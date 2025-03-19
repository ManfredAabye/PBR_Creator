# PBR Creator 1.12 2025 by Manfred Zainhofer

# Installation Python 3.13.2 mit Add Python to PATH
# python --version
# pip install numpy Pillow pygltflib scipy

# Program erstellen
# pip install pyinstaller
# pyinstaller --onefile PBR_Creator109.py
# pyinstaller --onefile --windowed PBR_Creator109.py

import numpy as np
from PIL import Image
from pygltflib import GLTF2, Material, PbrMetallicRoughness, TextureInfo, Image as GLTFImage, Buffer, BufferView, Accessor, Mesh, Node, Scene, Primitive, Sampler
import tkinter as tk
from tkinter import filedialog
import os
from scipy.ndimage import sobel
import base64
import json
import re

# 0. Konfiguration laden
def load_config(config_path="CreatorConfig.json"):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# 1. Textur laden
def load_texture(path):
    return np.array(Image.open(path).convert("RGB"))

# 2. Normal Map aus einer Höhenkarte generieren
def generate_normal_map(height_map, config):
    height_scale = config.get("height_scale", 1.0)
    intensity = config.get("intensity", 1.0)
    level = config.get("level", 1.0)
    blur_sharp = config.get("blur_sharp", "none")
    invert = config.get("invert", False)

    # In Graustufen konvertieren
    height_map = height_map.mean(axis=-1)

    # Blur/Sharp anwenden
    if blur_sharp == "blur":
        height_map = sobel(height_map, mode='constant')  # Beispiel unscharf
    elif blur_sharp == "sharp":
        height_map = height_map * 1.5  # Schärfen simulieren

    # Kontrast (Level)
    height_map = np.clip(height_map * level, 0, 255)

    # Gradienten berechnen
    dx = sobel(height_map, axis=1, mode='constant') * height_scale
    dy = sobel(height_map, axis=0, mode='constant') * height_scale
    dz = np.sqrt(1 - np.clip(dx**2 + dy**2, 0, 1))

    # Normal Map erstellen
    normal_map = np.stack([-dx, -dy, dz], axis=-1)
    normal_map = (normal_map + 1) * 127.5 * intensity

    # Invertieren
    if invert:
        normal_map[:, :, :2] = 255 - normal_map[:, :, :2]

    return normal_map.astype(np.uint8)


# 3. Roughness Map aus der Base Color Map generieren
def generate_roughness_map(base_color, config):
    invert = config.get("invert", True)
    contrast = config.get("contrast", 1.5)
    intensity = config.get("intensity", 1.0)
    blur_sharp = config.get("blur_sharp", "none")

    # In Graustufen konvertieren
    grayscale = base_color.mean(axis=-1)

    # Blur/Sharp anwenden
    if blur_sharp == "blur":
        grayscale = sobel(grayscale, mode='constant')
    elif blur_sharp == "sharp":
        grayscale = grayscale * 1.5

    # Kontrast und Intensität
    roughness_map = (255 - grayscale) if invert else grayscale
    roughness_map = np.clip(roughness_map * contrast * intensity, 0, 255)

    return np.stack([roughness_map] * 3, axis=-1).astype(np.uint8)


# 4. Occlusion Map aus der Base Color Map generieren
def generate_occlusion_map(base_color, config):
    invert = config.get("invert", True)
    strength = config.get("strength", 0.8)
    blur_sharp = config.get("blur_sharp", "none")
    intensity = config.get("intensity", 1.0)

    # In Graustufen konvertieren
    grayscale = base_color.mean(axis=-1)

    # Blur/Sharp anwenden
    if blur_sharp == "blur":
        grayscale = sobel(grayscale, mode='constant')
    elif blur_sharp == "sharp":
        grayscale = grayscale * 1.5

    # Anwendung von Stärke und Intensität
    occlusion_map = (255 - grayscale) if invert else grayscale
    occlusion_map = np.clip(occlusion_map * strength * intensity, 0, 255)

    return np.stack([occlusion_map] * 3, axis=-1).astype(np.uint8)


# 5. Metallic Map (standardmäßig nicht-metallisch)
def generate_metallic_map(base_color, config):
    default_value = config.get("default_value", 0.0)
    mask = config.get("mask", None)
    height, width, _ = base_color.shape

    # Standardmäßig keine Metallwerte
    metallic_map = np.full((height, width, 3), int(default_value * 255), dtype=np.uint8)

    # Maskenanwendung
    if mask is not None:
        metallic_map = np.where(mask, metallic_map, 0)

    return metallic_map


# 6. Emission Map (standardmäßig keine Emission)
def generate_emission_map(base_color, config):
    default_color = config.get("default_color", [0, 0, 0])
    intensity = config.get("intensity", 1.0)
    mask = config.get("mask", None)
    height, width, _ = base_color.shape

    # Standardfarbe und Intensität
    emission_map = np.full((height, width, 3), default_color, dtype=np.uint8)
    emission_map = np.clip(emission_map * intensity, 0, 255)

    # Maskenanwendung
    if mask is not None:
        emission_map = np.where(mask, emission_map, 0)

    return emission_map.astype(np.uint8)


# 7. Alpha Map (standardmäßig undurchsichtig)
def generate_alpha_map(base_color, config):
    default_value = config.get("default_value", 1.0)
    opacity = config.get("opacity", 1.0)
    mask = config.get("mask", None)
    height, width, _ = base_color.shape

    # Deckkraft anwenden
    alpha_map = np.full((height, width, 3), int(default_value * 255 * opacity), dtype=np.uint8)

    # Maskenanwendung
    if mask is not None:
        alpha_map = np.where(mask, alpha_map, 0)

    return alpha_map


# 8. Texturen anhand des Dateinamens suchen
def find_texture_files(base_path):
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]

    # Mögliche Suffixe für die Texturen
    ext_normals = ["_normal", "_norm", "_nrml", "_nrm", "_nor", "_normalmap", "_normals", "_normap", "_bump", "_n", "_normalgl", "_normaldx", "_tsn", "_wsn", "_normal-ogl"]
    ext_occlusion = ["_ambient", "_occlusion", "_ao", "_ambientocclusion", "_occl", "_ambocc", "_amb", "_occlmap"]
    ext_metallic = ["_metallic", "_metalness", "_mtl", "_metal", "_met", "_metalmap", "_metallicmap", "_metalnessmap"]
    ext_roughness = ["_roughness", "_rough", "_roug", "_rgh", "_roughmap", "_roughnessmap", "_rghmap"]
    ext_emission = ["_emission", "_emiss", "_emit", "_glow", "_illum", "_light", "_emissionmap", "_emissivemap"]
    ext_alpha = ["_alpha", "_transparency", "_opacity", "_mask", "_cutout", "_alphaMap", "_transparencymap", "_opac"]

    # Funktion zum Suchen von Texturen
    def find_texture(suffixes):
        for suffix in suffixes:
            texture_path = os.path.join(base_dir, f"{base_name}{suffix}.png")
            if os.path.exists(texture_path):
                return texture_path
        return None

    # Texturen suchen
    normal_map_path = find_texture(ext_normals)
    occlusion_map_path = find_texture(ext_occlusion)
    metallic_map_path = find_texture(ext_metallic)
    roughness_map_path = find_texture(ext_roughness)
    emission_map_path = find_texture(ext_emission)
    alpha_map_path = find_texture(ext_alpha)

    return {
        "normal_map": normal_map_path,
        "occlusion_map": occlusion_map_path,
        "metallic_map": metallic_map_path,
        "roughness_map": roughness_map_path,
        "emission_map": emission_map_path,
        "alpha_map": alpha_map_path
    }

# 9. GLTF 2.0-Datei erstellen
def create_gltf(base_texture, normal_map, occlusion_map, metallic_map, roughness_map, emission_map, alpha_map, output_path):
    gltf = GLTF2()
    gltf.asset = {"generator": "OpenManniLand", "version": "2.0"}

    # Verzeichnis für Texturen erstellen
    output_dir = os.path.splitext(output_path)[0]  # Verzeichnisname = GLTF-Dateiname ohne Endung
    os.makedirs(output_dir, exist_ok=True)

    # Texturen als externe Dateien speichern
    def save_texture(data, name):
        texture_path = os.path.join(output_dir, f"{name}.png")
        Image.fromarray(data).save(texture_path)
        return texture_path

    # Texturen speichern und GLTF-Images hinzufügen
    def add_image(texture_path):
        image = GLTFImage()
        image.uri = os.path.relpath(texture_path, os.path.dirname(output_path))  # Relativer Pfad
        gltf.images.append(image)
        return len(gltf.images) - 1

    base_texture_path = save_texture(base_texture, "TestMaterial_col")
    normal_map_path = save_texture(normal_map, "TestMaterial_nrm")
    metallic_roughness_path = save_texture(roughness_map, "TestMaterial_orm")
    emission_map_path = save_texture(emission_map, "TestMaterial_emission")

    base_texture_idx = add_image(base_texture_path)
    normal_map_idx = add_image(normal_map_path)
    metallic_roughness_idx = add_image(metallic_roughness_path)
    emission_map_idx = add_image(emission_map_path)

    # Sampler erstellen
    sampler = Sampler()
    sampler.magFilter = 9729
    sampler.minFilter = 9987
    sampler.wrapS = 33648
    sampler.wrapT = 33648
    gltf.samplers.append(sampler)

    # Texturen erstellen
    gltf.textures.extend([
        {"source": normal_map_idx, "sampler": 0},
        {"source": base_texture_idx, "sampler": 0},
        {"source": metallic_roughness_idx, "sampler": 0},
        {"source": emission_map_idx, "sampler": 0}
    ])

    # Material erstellen
    material = Material(
        doubleSided=False,
        name="TestMaterial",
        normalTexture=TextureInfo(index=0),
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=1),
            metallicFactor=0.0,
            metallicRoughnessTexture=TextureInfo(index=2)
        ),
        occlusionTexture=TextureInfo(index=2),
        emissiveTexture=TextureInfo(index=3),
        emissiveFactor=[1.0, 1.0, 1.0]
    )
    gltf.materials.append(material)

    # Geometrie erstellen (ein einfaches Quad)
    vertices = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0]
    ], dtype=np.float32)

    tex_coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint16)

    # Buffer erstellen
    buffer_data = np.concatenate([indices, vertices.flatten(), tex_coords.flatten()])
    buffer = Buffer()
    buffer.uri = "data:application/gltf-buffer;base64," + base64.b64encode(buffer_data.tobytes()).decode("utf-8")
    buffer.byteLength = len(buffer_data.tobytes())
    gltf.buffers.append(buffer)

    # BufferViews erstellen
    buffer_view_indices = BufferView()
    buffer_view_indices.buffer = 0
    buffer_view_indices.byteOffset = 0
    buffer_view_indices.byteLength = indices.nbytes
    buffer_view_indices.target = 34963  # ELEMENT_ARRAY_BUFFER
    gltf.bufferViews.append(buffer_view_indices)

    buffer_view_vertices = BufferView()
    buffer_view_vertices.buffer = 0
    buffer_view_vertices.byteOffset = indices.nbytes
    buffer_view_vertices.byteLength = vertices.nbytes
    buffer_view_vertices.byteStride = 12
    buffer_view_vertices.target = 34962  # ARRAY_BUFFER
    gltf.bufferViews.append(buffer_view_vertices)

    buffer_view_texcoords = BufferView()
    buffer_view_texcoords.buffer = 0
    buffer_view_texcoords.byteOffset = indices.nbytes + vertices.nbytes
    buffer_view_texcoords.byteLength = tex_coords.nbytes
    buffer_view_texcoords.byteStride = 8
    buffer_view_texcoords.target = 34962  # ARRAY_BUFFER
    gltf.bufferViews.append(buffer_view_texcoords)

    # Accessors erstellen
    accessor_indices = Accessor()
    accessor_indices.bufferView = 0
    accessor_indices.byteOffset = 0
    accessor_indices.componentType = 5123  # UNSIGNED_SHORT
    accessor_indices.count = len(indices)
    accessor_indices.type = "SCALAR"
    accessor_indices.max = [3]
    accessor_indices.min = [0]
    gltf.accessors.append(accessor_indices)

    accessor_vertices = Accessor()
    accessor_vertices.bufferView = 1
    accessor_vertices.byteOffset = 0
    accessor_vertices.componentType = 5126  # FLOAT
    accessor_vertices.count = len(vertices)
    accessor_vertices.type = "VEC3"
    accessor_vertices.max = [1.0, 1.0, 0.0]
    accessor_vertices.min = [-1.0, -1.0, 0.0]
    gltf.accessors.append(accessor_vertices)

    accessor_texcoords = Accessor()
    accessor_texcoords.bufferView = 2
    accessor_texcoords.byteOffset = 0
    accessor_texcoords.componentType = 5126  # FLOAT
    accessor_texcoords.count = len(tex_coords)
    accessor_texcoords.type = "VEC2"
    accessor_texcoords.max = [1.0, 1.0]
    accessor_texcoords.min = [0.0, 0.0]
    gltf.accessors.append(accessor_texcoords)

    # Mesh erstellen
    mesh = Mesh()
    mesh.primitives.append(Primitive(
        attributes={"POSITION": 1, "TEXCOORD_0": 2},
        indices=0,
        material=0
    ))
    gltf.meshes.append(mesh)

    # Node erstellen
    node = Node()
    node.mesh = 0
    gltf.nodes.append(node)

    # Scene erstellen
    scene = Scene()
    scene.nodes.append(0)
    gltf.scenes.append(scene)
    gltf.scene = 0

    # GLTF-Datei speichern
    gltf.save(output_path)
    print(f"GLTF-Datei gespeichert: {output_path}")
    print(f"Texturen gespeichert in: {output_dir}")

# 10. Hauptfunktion mit Tkinter
def main():
    # Konfiguration laden
    config = load_config("CreatoConfig110.json")

    # Tkinter-Fenster erstellen
    root = tk.Tk()
    root.withdraw()  # Hauptfenster verstecken

    # Textur auswählen
    texture_path = filedialog.askopenfilename(
        title="Textur auswählen",
        filetypes=[("PNG-Dateien", "*.png"), ("Alle Dateien", "*.*")]
    )
    if not texture_path:
        print("Keine Textur ausgewählt. Abbruch.")
        return

    # Speicherort für GLTF-Datei auswählen
    output_path = filedialog.asksaveasfilename(
        title="GLTF-Datei speichern",
        defaultextension=".gltf",
        filetypes=[("GLTF-Dateien", "*.gltf"), ("Alle Dateien", "*.*")]
    )
    if not output_path:
        print("Kein Speicherort ausgewählt. Abbruch.")
        return

    # Textur laden
    base_texture = load_texture(texture_path)

    # Texturen anhand des Dateinamens suchen
    texture_files = find_texture_files(texture_path)

    # PBR-Karten generieren oder vorhandene Texturen laden
    normal_map = load_texture(texture_files["normal_map"]) if texture_files["normal_map"] else generate_normal_map(base_texture, config.get("normal_map", {}))
    roughness_map = load_texture(texture_files["roughness_map"]) if texture_files["roughness_map"] else generate_roughness_map(base_texture, config.get("roughness_map", {}))
    occlusion_map = load_texture(texture_files["occlusion_map"]) if texture_files["occlusion_map"] else generate_occlusion_map(base_texture, config.get("occlusion_map", {}))
    metallic_map = load_texture(texture_files["metallic_map"]) if texture_files["metallic_map"] else generate_metallic_map(base_texture, config.get("metallic_map", {}))
    emission_map = load_texture(texture_files["emission_map"]) if texture_files["emission_map"] else generate_emission_map(base_texture, config.get("emission_map", {}))
    alpha_map = load_texture(texture_files["alpha_map"]) if texture_files["alpha_map"] else generate_alpha_map(base_texture, config.get("alpha_map", {}))

    # GLTF-Datei erstellen
    create_gltf(base_texture, normal_map, occlusion_map, metallic_map, roughness_map, emission_map, alpha_map, output_path)

if __name__ == "__main__":
    main()