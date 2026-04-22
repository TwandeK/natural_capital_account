# src/utils.py

# --- 1. Standaard Python Libraries ---
import os
import json
import logging
import requests
import zipfile
import subprocess
from pathlib import Path

# --- 2. Data & Wiskunde Libraries ---
import numpy as np
import pandas as pd

# --- 3. Ruimtelijke & Geografische Libraries ---
import geopandas as gpd
import fiona
import geojson

import rasterio as rio
from rasterio.mask import mask
from rasterio.plot import show, show_hist
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.windows import Window, from_bounds
from rasterio.enums import Resampling
from rasterio import features, windows
from rasterio.features import geometry_mask

# --- 4. Beeldverwerking (Image Processing) ---
from scipy.ndimage import gaussian_filter, generic_filter, uniform_filter, median_filter
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk


# --- 5. Visualisatie ---
import matplotlib.pyplot as plt
import seaborn as sns


def gemeente_of_interest_geodf(gpkg_path: Path, layer_name: str | None, gemeente: str, output_path: Path) -> gpd.GeoDataFrame:
    """
    Haalt een specifieke gemeente uit een GeoPackage.\n
    layername: ('gemeenten', 'provincies', 'landsgrens')\n
    gemeente: (volledige naam van je gemeente)\n
    Slaat deze op indien nieuw, of laadt deze direct in als het bestand al bestaat.
    """
    # 1. Controleer of de output al in je 'processed' map staat
    if output_path.exists():
        print(f"Data gevonden. Direct inladen vanaf: {output_path}")
        return gpd.read_file(output_path)
    
    # 2. Zo niet, voer de zware berekening uit
    print("Nieuwe data genereren...")
    layers = fiona.listlayers(gpkg_path)

    if layer_name is None:
        layer_name = layers[0] 
        print(f'Geen laag gespecificeerd. Default: {layer_name}')
    
    if layer_name not in layers:
        raise ValueError(f"Layer '{layer_name}' not found. Available layers: {layers}")
    
    gdf_layer = gpd.read_file(gpkg_path, layer=layer_name)
    chosen_gemeente = gdf_layer[gdf_layer['gemeentenaam'] == gemeente].copy()
    chosen_gemeente = chosen_gemeente.to_crs('EPSG:28992')

    # 3. Opslaan naar schijf voor de volgende keer
    chosen_gemeente.to_file(output_path)
    print(f"Nieuw bestand opgeslagen op: {output_path}")

    # 4. Return het object zodat je direct kunt doorrekenen
    return chosen_gemeente


def process_laz_to_buildings_tif(input_dir: Path, output_dir: Path, resolution: str = "0.25") -> list:
    """
    Verwerkt alle .laz bestanden in de input map via PDAL.\n
    Filtert uitsluitend op gebouwen (Classification[6:6]) en exporteert als .tif.\n
    **Output: {bestandsnaam}__gebouwen.tif**
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Zoek alle .laz bestanden (hoofdletterongevoelig)
    laz_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == '.laz']
    
    if not laz_files:
        print(f"Geen .laz bestanden gevonden in: {input_dir}")
        return []
    
    expected_outputs = [output_dir / f"{pc.stem}_gebouwen.tif" for pc in laz_files]
    if all(out_file.exists() for out_file in expected_outputs):
        print("Alle gebouwen-rasters bestaan al. Berekening overgeslagen.")
        return expected_outputs

    print(f"Start verwerking van {len(laz_files)} puntenwolken (Alleen Gebouwen)...")
    output_files = []

    for pc in laz_files:
        base_name = pc.stem 
        output_file = output_dir / f"{base_name}_gebouwen.tif"
        output_files.append(output_file)
        
        if output_file.exists():
            print(f" -> Overslaan: {base_name} (Bestand bestaat al)")
            continue
            
        print(f" -> Bezig met: {base_name}")

        # Pipeline definitie
        pipeline = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": str(pc),
                    "nosrs": True,                   # Negeer de ontbrekende WKT-vlag in AHN5 data
                    "override_srs": "EPSG:28992"     # Forceer het Amersfoort/RD New coördinatensysteem
                },
                {
                    "type": "filters.range",
                    "limits": "Classification[6:6]"  # Alleen gebouwen
                },
                {
                    "type": "writers.gdal",
                    "filename": str(output_file),
                    "resolution": resolution,
                    "output_type": "max",  
                    "window_size": 0                 # Geen gaten opvullen, strakke grenzen
                }
            ]
        }
        
        process = subprocess.run(
            ["pdal", "pipeline", "--stdin"],
            input=json.dumps(pipeline),
            capture_output=True,
            text=True
        )

        if process.returncode == 0:
            print(f"    Gereed.")
        else:
            print(f"    Fout:\n{process.stderr}")

    print("Alle LAZ-bestanden verwerkt.")
    return output_files



def merge_raster_to_mosaic(input_data, output_file, search_pattern="*.tif", nodata=-9999.0):
    """
    Memory-safe merging via GDAL VRT. 
    Kiest dynamisch de juiste compressie-predictor op basis van het datatype.
    """
    out_path = Path(output_file)
    if out_path.exists():
        print(f"Bestand bestaat al. Berekening overgeslagen: {out_path.name}")
        return out_path

    if isinstance(input_data, list):
        paths_to_merge = [str(p) for p in input_data]
    else:
        paths_to_merge = [str(p) for p in Path(input_data).glob(search_pattern)]

    if not paths_to_merge:
        print("Geen bestanden gevonden om te mergen!")
        return None

    # --- DE FIX: Bepaal dynamisch de juiste predictor ---
    with rio.open(paths_to_merge[0]) as src:
        is_float = 'float' in str(src.dtypes[0])
        predictor = "3" if is_float else "2"

    vrt_path = out_path.with_suffix('.vrt')
    
    print(f"Bouwen van virtueel mozaïek (VRT) voor {len(paths_to_merge)} bestanden (Low RAM)...")
    
    subprocess.run([
        "gdalbuildvrt", 
        "-srcnodata", str(nodata), 
        "-vrtnodata", str(nodata), 
        str(vrt_path)
    ] + paths_to_merge, check=True)

    print(f"Schrijven naar definitieve TIF (Compressie Predictor: {predictor}). Dit kan even duren...")
    
    subprocess.run([
        "gdal_translate",
        "-of", "GTiff",
        "-co", "COMPRESS=DEFLATE",
        "-co", f"PREDICTOR={predictor}",
        "-co", "ZLEVEL=6",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=YES",
        "-co", "BLOCKXSIZE=512",
        "-co", "BLOCKYSIZE=512",
        str(vrt_path),
        str(out_path)
    ], check=True)

    print("Overviews genereren voor snelle visualisatie...")
    subprocess.run(["gdaladdo", "-r", "nearest", str(out_path), "2", "4", "8", "16", "32"], check=True)

    vrt_path.unlink()
    
    print(f"Mosaic succesvol aangemaakt: {out_path.name}")
    return out_path


def resample_raster(input_path, output_path, target_res=0.5):
    """
    Past de resolutie van een raster aan naar een gewenste pixelgrootte (target_res).
    """
    out_path = Path(output_path)
    if out_path.exists():
        print(f"Bestand bestaat al. Berekening overgeslagen, pad geretourneerd: {out_path}")
        return out_path

    with rio.open(input_path) as src:
        scale_factor = src.res[0] / target_res

        #Bereken de nieuwe dimensies
        new_height = int(src.height * scale_factor)
        new_width = int(src.width * scale_factor)

        #De data inlezen en herschalen
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling = Resampling.average if scale_factor < 1 else Resampling.bilinear
        )

        #georeferentie bijwerken. Stapgrootte bijwerken, anders verschuift kaart
        new_transform = src.transform * src.transform.scale(
            src.width / data.shape[-1],
            src.height / data.shape[-2]
        )

        #Metadata voorbereiden voor het nieuwe bestand.
        out_meta = src.meta.copy()
        out_meta.update({
                    "driver": "GTiff",
                    "BIGTIFF": "YES",
                    "height": new_height,
                    "width": new_width,
                    "transform": new_transform,
                    "compress": "lzw", # LZW compressie houdt je bestanden klein zonder kwaliteitsverlies
                    "tiled": True
                })
        
        # nieuwe bestand opslaan.
        with rio.open(out_path, "w", **out_meta) as dst:
            dst.write(data)
    
    print(f"Gereed: {out_path.name} is nu {target_res}m resolutie.")
    return out_path


def check_resampling(original_path, resampled_path, window_off=(6000, 3000), size_m=10):
    """
    VISUELE VALIDATIE VAN RESAMPLING:\n 
    Vergelijkt een specifiek gebied tussen het origineel en de resampled versie.
    size_m: grootte van de subset in meters.
    """
    with rio.open(original_path) as src_high, rio.open(resampled_path) as src_low:
        
        # 1. Bereken venster voor hoge resolutie (bijv. 8cm of 25cm)
        # We pakken een gebied van 'size_m' meter
        win_high = Window(window_off[0], window_off[1], 
                          int(size_m / src_high.res[0]), 
                          int(size_m / src_high.res[1]))
        
        # 2. Bereken venster voor lage resolutie (0.5m)
        # We moeten de offset schalen omdat er minder pixels zijn!
        scale = src_low.res[0] / src_high.res[0]
        win_low = Window(int(window_off[0] / scale), 
                         int(window_off[1] / scale), 
                         int(size_m / src_low.res[0]), 
                         int(size_m / src_low.res[1]))

        # Data inlezen
        data_high = src_high.read(1, window=win_high)
        data_low = src_low.read(1, window=win_low)

        # Plotten
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        show(data_high, ax=ax1, cmap='RdYlGn', title=f'Origineel ({src_high.res[0]}m)')
        show(data_low, ax=ax2, cmap='RdYlGn', title=f'Resampled ({src_low.res[0]}m)')

        plt.show()

        # Statistiek check
        print(f"--- Statistiek Vergelijking (Subset) ---")
        print(f"Origineel - Min: {data_high.min():.2f}, Max: {data_high.max():.2f}, Mean: {data_high.mean():.2f}")
        print(f"Resampled - Min: {data_low.min():.2f}, Max: {data_low.max():.2f}, Mean: {data_low.mean():.2f}")


def clip_raster_to_shape(input_raster_path, gemeente_geometry, output_name):
    """
    Knipt een raster bij op basis van een vector-polygoon (shapefile/geopackage).
    Uitgevoerd in memory-safe blokken (windows) om RAM-crashes te voorkomen.
    """
    out_path = Path(output_name)
    if out_path.exists():
        print(f" Bestand bestaat al: {out_path.name}")
        return out_path

    # 1. Geometrie inladen/kopiëren
    if isinstance(gemeente_geometry, (str, Path)):
        gdf = gpd.read_file(gemeente_geometry)
    else:
        gdf = gemeente_geometry.copy()

    with rio.open(input_raster_path) as src:
        # Check projectie
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # 2. Bepaal de Bounding Box (buitenste randen) van de gemeente
        minx, miny, maxx, maxy = gdf.total_bounds
        
        # 3. Bereken het corresponderende 'venster' in de enorme bron-TIF
        crop_window = from_bounds(minx, miny, maxx, maxy, src.transform)
        # Rond af naar hele pixels om sub-pixel verschuivingen te voorkomen
        crop_window = crop_window.round_offsets().round_lengths()
        crop_transform = src.window_transform(crop_window)

        # 4. Metadata instellen (Compressie & Tiling)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": crop_window.height,
            "width": crop_window.width,
            "transform": crop_transform,
            "nodata": -9999,
            "compress": "deflate",
            "predictor": 3,
            "zlevel": 6,
            "BIGTIFF": "YES",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512
        })

        # 5. Blok voor blok inlezen, maskeren en wegschrijven
        with rio.open(out_path, "w", **out_meta) as dst:
            dst.descriptions = src.descriptions
            
            windows_list = list(dst.block_windows(1))
            totaal_blokken = len(windows_list)
            print(f"Start memory-safe clipping ({totaal_blokken} blokken)...")

            for i, (ij, dst_window) in enumerate(windows_list):
                
                # A. Bereken waar dit kleine blokje zich bevindt in de grote originele TIF
                src_window = Window(
                    col_off=crop_window.col_off + dst_window.col_off,
                    row_off=crop_window.row_off + dst_window.row_off,
                    width=dst_window.width,
                    height=dst_window.height
                )
                
                # B. Lees uitsluitend dit kleine blokje in (Kost ~10MB RAM)
                data = src.read(window=src_window)
                
                # C. Maak een masker van de gemeentegrens (voor dit specifieke blokje)
                win_transform = src.window_transform(src_window)
                
                # geometry_mask geeft 'True' voor pixels die BUITEN de gemeente vallen
                mask_buiten_grens = geometry_mask(
                    geometries=gdf.geometry,
                    out_shape=(dst_window.height, dst_window.width),
                    transform=win_transform,
                    invert=False 
                )
                
                # D. Zet alle pixels buiten de gemeentegrens op -9999 (NoData)
                data[:, mask_buiten_grens] = -9999
                
                # E. Schrijf dit blokje weg
                dst.write(data, window=dst_window)
                
                # Voortgang printen
                if i % 25 == 0 or i == totaal_blokken - 1:
                    print(f"Voortgang: {i + 1}/{totaal_blokken} blokken", end='\r')

    print(f"\n Clipping succesvol afgerond.")
    return out_path


def create_feature_stack_windowed(output_path, ndsm_path, ndsm_median_path, ndvi_path, 
                                 rgb_path, infra_path, entropy_path, texture_path, 
                                 exg_path, tpi_path, roughness_path, gebouwen_path):
    
    out_path = Path(output_path)
    if out_path.exists():
        print(f"Bestand bestaat al. Berekening overgeslagen: {out_path}")
        return out_path

    with rio.open(ndvi_path) as src_master, \
         rio.open(rgb_path) as src_rgb, \
         rio.open(infra_path) as src_nir, \
         rio.open(entropy_path) as src_ent, \
         rio.open(texture_path) as src_tex, \
         rio.open(exg_path) as src_exg, \
         rio.open(tpi_path) as src_tpi, \
         rio.open(roughness_path) as src_rough, \
         rio.open(ndsm_path) as src_ndsm, \
         rio.open(ndsm_median_path) as src_median, \
         rio.open(gebouwen_path) as src_geb:

        master_meta = src_master.meta.copy()
        
        master_meta.update({
            "count": 13,
            "dtype": 'float32',
            "compress": 'deflate',  
            "predictor": 3,         
            "zlevel": 6,            
            "BIGTIFF": "YES",
            "tiled": True,
            "blockxsize": 512,      
            "blockysize": 512,
            "nodata": -9999         
        })

        print(f"Start windowed processing voor {out_path.name}")

        with rio.open(out_path, "w", **master_meta) as dst:
            dst.descriptions = ['nDSM', 'nDSM_Median', 'NDVI', 'Red', 'Green', 'Blue', 'NIR', 'entropy', 'texture', 'ExG', 'TPI', 'Roughness', 'gebouwen']
            
            for ij, window in src_master.block_windows():
                win_h, win_w = window.height, window.width
                chunk = np.zeros((13, win_h, win_w), dtype='float32')

                chunk[2] = src_master.read(1, window=window).astype('float32') 
                chunk[3:6] = src_rgb.read(window=window).astype('float32')     
                chunk[6] = src_nir.read(1, window=window).astype('float32')    
                chunk[7] = src_ent.read(1, window=window).astype('float32')    
                chunk[8] = src_tex.read(1, window=window).astype('float32')
                chunk[9] = src_exg.read(1, window=window).astype('float32')
                chunk[10] = src_tpi.read(1, window=window).astype('float32')
                chunk[11] = src_rough.read(1, window=window).astype('float32')

                win_bounds = src_master.window_bounds(window)
                for src_ext, idx in [(src_ndsm, 0), (src_median, 1), (src_geb, 12)]:
                    ext_window = src_ext.window(*win_bounds)
                    chunk[idx] = src_ext.read(1, window=ext_window, out_shape=(win_h, win_w)).astype('float32')

                gebouw_masker = chunk[12] > 0
                chunk[0:12, gebouw_masker] = -9999
                np.nan_to_num(chunk, copy=False, nan=-9999)

                dst.write(chunk, window=window)

    print(f"Stack opgeslagen: {out_path.name}")
    return out_path


def create_texture_entropy_layers(rgb_path, entropy_output_path, contrast_output_path, window_radius=1):
    """
    Genereert entropie- en contrastlagen (std) met windowed processing.
    Geoptimaliseerd voor grote rasters (>4GB) en floating-point compressie.
    """
    ent_path, con_path = Path(entropy_output_path), Path(contrast_output_path)
    
    # Skip als bestanden al bestaan
    if ent_path.exists() and con_path.exists():
        return ent_path, con_path

    pad = window_radius + 2 
    
    with rio.open(rgb_path) as src:
        # 1. Metadata voorbereiden met Predictor 3 en Deflate
        profile = src.profile.copy()
        profile.update({
            "dtype": rio.float32,
            "count": 1,
            "nodata": -9999,
            "compress": "deflate",  # Sterkere compressie voor texturen
            "predictor": 3,         # Floating-point predictor: cruciaal voor float32 besparing!
            "zlevel": 6,            # Balans tussen snelheid en compressie
            "tiled": True,
            "blockxsize": 512,      # Grotere blokken voor efficiëntere I/O op SSD
            "blockysize": 512,
            "BIGTIFF": "YES"        # Verplicht voor 25cm regio-mosaics
        })
        
        windows_list = list(src.block_windows())
        total = len(windows_list)
        
        with rio.open(ent_path, 'w', **profile) as dst_ent, \
             rio.open(con_path, 'w', **profile) as dst_std:
            
            print(f"Start textuurberekening ({total} blokken)...")
            
            for i, (ij, window) in enumerate(windows_list):
                # Bereken padded window voor context rondom de randen
                read_win = Window(
                    window.col_off - pad, window.row_off - pad,
                    window.width + 2*pad, window.height + 2*pad
                ).intersection(Window(0, 0, src.width, src.height))
                
                # Lees RGB en converteer naar grijs (uint8 voor entropie)
                rgb_block = src.read([1, 2, 3], window=read_win)
                # Transpose van (Bands, H, W) naar (H, W, Bands) voor skimage
                gray_block = (rgb2gray(np.transpose(rgb_block, (1, 2, 0))) * 255).astype(np.uint8)
                
                # --- Berekening 1: Entropie (Lokale complexiteit) ---
                ent_block = entropy(gray_block, disk(window_radius))
                
                # --- Berekening 2: Contrast (Standaard Deviatie via snelle filter) ---
                f_block = gray_block.astype(np.float32)
                # Gebruik window_radius*2+1 om een oneven diameter te krijgen
                size = window_radius * 2 + 1
                c1 = uniform_filter(f_block, size=size)
                c2 = uniform_filter(f_block * f_block, size=size)
                std_block = np.sqrt(np.maximum(c2 - c1**2, 0))
                
                # --- Padding verwijderen en schrijven ---
                r_start = window.row_off - read_win.row_off
                c_start = window.col_off - read_win.col_off
                
                out_slice = (slice(r_start, r_start + window.height), 
                             slice(c_start, c_start + window.width))
                
                dst_ent.write(ent_block[out_slice].astype(np.float32), 1, window=window)
                dst_std.write(std_block[out_slice].astype(np.float32), 1, window=window)
                
                if i % 100 == 0:
                    print(f"Voortgang: {i}/{total} blokken voltooid", end='\r')

    print(f"\nTextuurlagen succesvol opgeslagen in {ent_path.parent}")
    return ent_path, con_path


def maak_training_dataset_tiled(raster_path, vector_path, output_csv_path):
    """
    Zet een satellietbeeld en een vectorbestand om naar een CSV voor Machine Learning.
    Verwerkt de data in kleine blokjes (tiles) om geheugenproblemen te voorkomen.
    """
    out_csv = Path(output_csv_path)
    
    # 1: Voorbereiding bestanden
    if out_csv.exists():
        print(f"Dataset bestaat al. Berekening overgeslagen, pad geretourneerd: {out_csv}")
        return out_csv

    vector_data = gpd.read_file(vector_path) #de gelabelde polygonen

    # 2: Labels van polygonen omzetten naar getallen
    kolom_namen = ['boom', 'gras', 'water_weg']
    name_to_num = {naam: i+1 for i, naam in enumerate(kolom_namen)}
    num_to_name = {v: k for k, v in name_to_num.items()}
    
    vector_data['label_naam'] = vector_data[kolom_namen].astype(float).idxmax(axis=1) #pakt de naam van de kolom die de hoogste waarde heef
    vector_data['label_nummer'] = vector_data['label_naam'].map(name_to_num) #converts text to the ID (name_to_num)
    
    print("Start met het verwerken van tegels (tiles)...")

    # 3: Satellietbeeld openen en verwerken
    with rio.open(raster_path) as satelliet_beeld:
        
        if vector_data.crs != satelliet_beeld.crs:
            print(f"Projecties verschillen. Vector wordt omgezet naar {satelliet_beeld.crs}")
            vector_data = vector_data.to_crs(satelliet_beeld.crs)
            
        band_namen = [satelliet_beeld.descriptions[i] or f"band_{i+1}" for i in range(satelliet_beeld.count)]
        
        is_eerste_blokje = True 

        # LOOP 
        for blok_index, venster in satelliet_beeld.block_windows(1):

            # A. in blokken laden
            venster_grenzen = windows.bounds(venster, satelliet_beeld.transform)
            xmin, ymin, xmax, ymax = venster_grenzen
            
            # B. zoek polygoon in blok
            polygonen_in_venster = vector_data.cx[xmin:xmax, ymin:ymax] #welke bomen zitten in dit venster?
            if polygonen_in_venster.empty:
                continue 

            # C. Rasteriseren: van de polygonen een grid maken met nummers (1=bomen 2=etc.)
            venster_transformatie = satelliet_beeld.window_transform(venster)
            vormen_en_labels = ((geom, waarde) for geom, waarde in zip(polygonen_in_venster.geometry, polygonen_in_venster['label_nummer']))
            
            label_masker = features.rasterize(
                shapes=vormen_en_labels,
                out_shape=(venster.height, venster.width),
                transform=venster_transformatie,
                fill=0,
                all_touched=True,
                dtype=rio.uint8
            )
            
            # D. Data Lezen
            pixel_data = satelliet_beeld.read(window=venster)
            
            # E. Plat slaan 3D (Height, Width, Bands) --> 2D (rows=Samples, columns=Features)
            aantal_banden = satelliet_beeld.count
            alle_pixel_waardes = pixel_data.reshape(aantal_banden, -1).T
            alle_pixel_labels = label_masker.reshape(-1)
                        
            # F. Haal de NoData waarde op. Vervang NoData door 0 in de hele array en Nan naar 0
            nodata_waarde = satelliet_beeld.nodata if satelliet_beeld.nodata is not None else -9999
            alle_pixel_waardes = alle_pixel_waardes.astype(float)
            alle_pixel_waardes[alle_pixel_waardes == nodata_waarde] = np.nan

            # G. Filteren. Maak mask om straks de <0 waardes weg te filteren.
            wel_data_masker = alle_pixel_labels > 0
            if wel_data_masker.sum() == 0:
                continue

            # Pas het filter toe. Bewaar de pixels waar label voor is (> 0).
            gefilterde_pixels = alle_pixel_waardes[wel_data_masker]
            gefilterde_labels = alle_pixel_labels[wel_data_masker]
            
            # H. Opslaan
            df_chunk = pd.DataFrame(gefilterde_pixels, columns=band_namen)
            df_chunk['label_id'] = gefilterde_labels
            df_chunk['label'] = df_chunk['label_id'].map(num_to_name)
            
            df_chunk.to_csv(out_csv, mode='a', header=is_eerste_blokje, index=False)
            is_eerste_blokje = False

    print(f"De dataset is opgeslagen als: {out_csv}")
    return out_csv


def gaussian_blur_filter(ndsm_input_path, gaussian_output_path, sigma=2, buffer=5):
    
    out_path = Path(gaussian_output_path)
    if out_path.exists():
        print(f"Bestand bestaat al. Berekening overgeslagen, pad geretourneerd: {out_path}")
        return out_path

    # 1. Lees DSM (Het Oppervlak / Ruw)
    with rio.open(ndsm_input_path) as src_ndsm:

        out_meta = src_ndsm.meta.copy()
        out_meta.update({
            "dtype": "float32",
            "compress": "lzw",
            "BIGTIFF": "YES",   
            "tiled": True,       
            "blockxsize": 256,
            "blockysize": 256
        }) 

        windows_list = list(src_ndsm.block_windows(1))
        totaal_blokken = len(windows_list)

        with rio.open(out_path, "w", **out_meta) as dst:
            for i, (block_index, venster) in enumerate(windows_list):

                padded_window = Window(
                    venster.col_off - buffer,
                    venster.row_off - buffer,
                    venster.width + 2*buffer,
                    venster.height + 2*buffer
                )

                img_padded = src_ndsm.read(1, window=padded_window, boundless=True, fill_value=0)
                img_padded[img_padded < -100] = 0
                img_padded = np.nan_to_num(img_padded, nan=0.0)

                blurred_padded = gaussian_filter(img_padded, sigma=sigma)
                blurred_padded[blurred_padded < 0] = 0

                final_block = blurred_padded[buffer:-buffer, buffer:-buffer]

                dst.write(final_block.astype('float32'), window=venster, indexes=1)

                if i % 100 == 0:
                    print(f"Blok {i+1}/{totaal_blokken} verwerkt...", end='\r')

    print(f"Succesvol opgeslagen: {out_path}")
    return out_path

def median_filter_ndsm(ndsm_input_path, median_output_path, filter_size=3, buffer=5):
    """
    Past een rand-behoudend median filter toe om ruis te verwijderen zonder
    valleien tussen bomen op te vullen of kleine pieken plat te slaan.
    """
    out_path = Path(median_output_path)
    if out_path.exists():
        print(f"Bestand bestaat al. Berekening overgeslagen, pad geretourneerd: {out_path}")
        return out_path

    # 1. Lees nDSM (Het Oppervlak)
    with rio.open(ndsm_input_path) as src_ndsm:

        out_meta = src_ndsm.meta.copy()
        
        # Upgrade compressie naar deflate + predictor 3 (optimaal voor float32)
        # Vergroot blokken voor snellere SSD I/O
        out_meta.update({
            "dtype": "float32",
            "compress": "deflate",
            "predictor": 3,
            "zlevel": 6,
            "BIGTIFF": "YES",   
            "tiled": True,       
            "blockxsize": 512,
            "blockysize": 512
        }) 

        windows_list = list(src_ndsm.block_windows(1))
        totaal_blokken = len(windows_list)

        with rio.open(out_path, "w", **out_meta) as dst:
            for i, (block_index, venster) in enumerate(windows_list):

                padded_window = Window(
                    venster.col_off - buffer,
                    venster.row_off - buffer,
                    venster.width + 2*buffer,
                    venster.height + 2*buffer
                )

                img_padded = src_ndsm.read(1, window=padded_window, boundless=True, fill_value=0)
                img_padded[img_padded < -100] = 0
                img_padded = np.nan_to_num(img_padded, nan=0.0)

                # --- DE FIX: MEDIAN FILTER ---
                filtered_padded = median_filter(img_padded, size=filter_size)
                filtered_padded[filtered_padded < 0] = 0

                final_block = filtered_padded[buffer:-buffer, buffer:-buffer]

                dst.write(final_block.astype('float32'), window=venster, indexes=1)

                if i % 100 == 0:
                    print(f"Blok {i+1}/{totaal_blokken} verwerkt...", end='\r')

    print(f"\nSuccesvol opgeslagen: {out_path}")
    return out_path

import rasterio as rio
from rasterio.windows import Window
import numpy as np
from pathlib import Path

def create_exg_layer(rgb_path, exg_output_path):
    """
    Berekent de Excess Green (ExG) index: (2 * Groen) - Rood - Blauw.
    Verwerkt in blokken om RAM-crashes op enorme rasters te voorkomen.
    """
    out_path = Path(exg_output_path)
    
    if out_path.exists():
        print(f"Bestand bestaat al. Berekening overgeslagen, variabele is 'exg_mosaic'.\n Pad:{out_path}")
        return out_path

    with rio.open(rgb_path) as src_rgb:
        # 1. Metadata voorbereiden voor efficiënte float32 opslag
        out_meta = src_rgb.meta.copy()
        out_meta.update({
            "dtype": "float32",
            "count": 1,
            "nodata": -9999,
            "compress": "deflate",  
            "predictor": 3,         
            "zlevel": 6,            
            "tiled": True,
            "blockxsize": 512,      
            "blockysize": 512,
            "BIGTIFF": "YES"        
        })
        
        windows_list = list(src_rgb.block_windows(1))
        total = len(windows_list)
        
        print(f"Start ExG berekening ({total} blokken)...")
        
        with rio.open(out_path, 'w', **out_meta) as dst:
            for i, (ij, window) in enumerate(windows_list):
                
                # 2. Lees RGB in (Band 1=Rood, 2=Groen, 3=Blauw)
                rgb_chunk = src_rgb.read([1, 2, 3], window=window).astype('float32')
                red, green, blue = rgb_chunk[0], rgb_chunk[1], rgb_chunk[2]
                
                # 3. ExG Berekening
                exg = (2 * green) - red - blue
                
                # 4. Nodata maskeren (aanname: originele lege pixels zijn 0 of specifieke nodata)
                nodata_val = src_rgb.nodata if src_rgb.nodata is not None else 0
                empty_mask = (red == nodata_val) & (green == nodata_val) & (blue == nodata_val)
                exg[empty_mask] = -9999
                
                # 5. Blok wegschrijven
                dst.write(exg, 1, window=window)
                
                if i % 100 == 0:
                    print(f"Voortgang: {i}/{total} blokken voltooid", end='\r')

    print(f"\nExG succesvol opgeslagen op:\n{out_path}")
    return out_path


def create_ndvi_layer(rgb_path, nir_path, ndvi_output_path):
    """
    Berekent de NDVI (Normalized Difference Vegetation Index).
    Verwerkt in memory-safe blokken (windows) en geoptimaliseerde float32 opslag.
    """
    out_path = Path(ndvi_output_path)
    
    if out_path.exists():
        print(f"Bestand bestaat al. Berekening overgeslagen, variabele is 'ndvi_mosaic'.\n Pad:{out_path}")
        return out_path

    # 1. Open beide bestanden tegelijk
    with rio.open(rgb_path) as src_rgb, rio.open(nir_path) as src_nir:
        
        # 2. Metadata voorbereiden voor efficiënte float32 opslag
        out_meta = src_rgb.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "nodata": -9999,
            "compress": "deflate",  # Beter dan lzw voor floats
            "predictor": 3,         
            "zlevel": 6,            
            "tiled": True,
            "blockxsize": 512,      
            "blockysize": 512,
            "BIGTIFF": "YES"        
        })
        
        windows_list = list(src_rgb.block_windows(1))
        total = len(windows_list)
        
        print(f"Start NDVI berekening ({total} blokken)...")
        
        with rio.open(out_path, 'w', **out_meta) as dst:
            for i, (ij, window) in enumerate(windows_list):
                
                # 3. Lees Rood (Band 1 uit RGB) en NIR (Band 1 uit Infra)
                red = src_rgb.read(1, window=window).astype('float32')
                nir = src_nir.read(1, window=window).astype('float32')
                
                # 4. NDVI Berekening (Vectorized met fout-afhandeling)
                with np.errstate(divide='ignore', invalid='ignore'):
                    noemer = nir + red
                    
                    # np.where voorkomt divide-by-zero en zet lege pixels direct op nodata
                    ndvi = np.where(noemer == 0, -9999, (nir - red) / noemer)

                # Nodata maskeren (extra check voor als originele data al nodata was)
                nodata_rgb = src_rgb.nodata if src_rgb.nodata is not None else 0
                nodata_nir = src_nir.nodata if src_nir.nodata is not None else 0
                empty_mask = (red == nodata_rgb) | (nir == nodata_nir)
                ndvi[empty_mask] = -9999
                
                # 5. Blok wegschrijven
                dst.write(ndvi, 1, window=window)
                
                if i % 100 == 0:
                    print(f"Voortgang: {i}/{total} blokken voltooid", end='\r')

    print(f"\nNDVI succesvol opgeslagen op:\n{out_path}")
    return out_path

import rasterio as rio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
import numpy as np
from pathlib import Path

def create_tpi_layer(ndsm_path, tpi_output_path, filter_size=5, buffer=5):
    """
    Berekent de Topographic Position Index (TPI).
    Vindt lokale pieken (kleine bomen) door hoogte te vergelijken met omgeving.
    Maakt gebruik van padding om naadloze overgangen tussen tegels te garanderen.
    """
    out_path = Path(tpi_output_path)
    if out_path.exists():
        print(f"Bestand bestaat al: {out_path.name}")
        return out_path

    with rio.open(ndsm_path) as src:
        out_meta = src.meta.copy()
        out_meta.update({
            "dtype": "float32", "count": 1, "nodata": -9999,
            "compress": "deflate", "predictor": 3, "zlevel": 6,
            "tiled": True, "blockxsize": 512, "blockysize": 512, "BIGTIFF": "YES"
        })

        windows_list = list(src.block_windows(1))
        total = len(windows_list)
        print(f"Start TPI berekening ({total} blokken)...")

        with rio.open(out_path, "w", **out_meta) as dst:
            for i, (ij, venster) in enumerate(windows_list):
                # Padded venster om edge-artifacts te voorkomen
                padded_window = Window(
                    venster.col_off - buffer, venster.row_off - buffer,
                    venster.width + 2*buffer, venster.height + 2*buffer
                )

                img_padded = src.read(1, window=padded_window, boundless=True, fill_value=0).astype('float32')
                img_padded = np.nan_to_num(img_padded, nan=0.0)

                # TPI Wiskunde: Huidige pixel minus gemiddelde van de omgeving
                mean_omgeving = uniform_filter(img_padded, size=filter_size)
                tpi_padded = img_padded - mean_omgeving

                # Randen afknippen en Nodata herstellen
                final_block = tpi_padded[buffer:-buffer, buffer:-buffer]
                nodata_val = src.nodata if src.nodata is not None else -9999
                
                # Check waar originele data (binnen het originele venster) leeg was
                orig_data = src.read(1, window=venster, boundless=True, fill_value=nodata_val)
                final_block[orig_data == nodata_val] = -9999

                dst.write(final_block, 1, window=venster)
                if i % 100 == 0:
                    print(f"Voortgang: {i}/{total} blokken", end='\r')

    print(f"\nTPI succesvol opgeslagen op:\n{out_path}")
    return out_path


def create_roughness_layer(ndsm_path, roughness_output_path, filter_size=3, buffer=5):
    """
    Berekent Hoogte-Ruwheid (Lokale standaarddeviatie).
    Onderscheidt chaotisch bladerdek van glad gras of strakke daken.
    """
    out_path = Path(roughness_output_path)
    if out_path.exists():
        print(f"Bestand bestaat al: {out_path.name}")
        return out_path

    with rio.open(ndsm_path) as src:
        out_meta = src.meta.copy()
        out_meta.update({
            "dtype": "float32", "count": 1, "nodata": -9999,
            "compress": "deflate", "predictor": 3, "zlevel": 6,
            "tiled": True, "blockxsize": 512, "blockysize": 512, "BIGTIFF": "YES"
        })

        windows_list = list(src.block_windows(1))
        total = len(windows_list)
        print(f"Start Roughness berekening ({total} blokken)...")

        with rio.open(out_path, "w", **out_meta) as dst:
            for i, (ij, venster) in enumerate(windows_list):
                padded_window = Window(
                    venster.col_off - buffer, venster.row_off - buffer,
                    venster.width + 2*buffer, venster.height + 2*buffer
                )

                img_padded = src.read(1, window=padded_window, boundless=True, fill_value=0).astype('float32')
                img_padded = np.nan_to_num(img_padded, nan=0.0)

                # Roughness Wiskunde: Snelle lokale standaarddeviatie
                mean_sq = uniform_filter(img_padded**2, size=filter_size)
                sq_mean = uniform_filter(img_padded, size=filter_size)**2
                rough_padded = np.sqrt(np.maximum(mean_sq - sq_mean, 0))

                # Randen afknippen
                final_block = rough_padded[buffer:-buffer, buffer:-buffer]
                nodata_val = src.nodata if src.nodata is not None else -9999
                
                orig_data = src.read(1, window=venster, boundless=True, fill_value=nodata_val)
                final_block[orig_data == nodata_val] = -9999

                dst.write(final_block, 1, window=venster)
                if i % 100 == 0:
                    print(f"Voortgang: {i}/{total} blokken", end='\r')

    print(f"\nRoughness succesvol opgeslagen op:\n{out_path}")
    return out_path