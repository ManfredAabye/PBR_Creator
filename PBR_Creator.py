import numpy as np
from PIL import Image
from pygltflib import GLTF2, Material, PbrMetallicRoughness, TextureInfo, Image as GLTFImage, Buffer, BufferView, Accessor, Mesh, Node, Scene, Primitive, Sampler
import tkinter as tk
from tkinter import filedialog
import os
from scipy.ndimage import sobel
import base64

# PBR Creator 1.09 2025 by Manfred Zainhofer

# Installation Python 3.13.2 mit Add Python to PATH
# pip install numpy Pillow pygltflib scipy

# Program erstellen
# pip install pyinstaller
# pyinstaller --onefile PBR_Creator109.py
# pyinstaller --onefile --windowed PBR_Creator109.py

# 1. Textur laden
def load_texture(path):
    return np.array(Image.open(path).convert("RGB"))

# 2. Normal Map aus einer Höhenkarte generieren
def generate_normal_map(height_map):
    height_map = height_map.mean(axis=-1)  # In Graustufen konvertieren
    dx = sobel(height_map, axis=1, mode='constant')
    dy = sobel(height_map, axis=0, mode='constant')
    dz = np.sqrt(1 - np.clip(dx**2 + dy**2, 0, 1))

    normal_map = np.stack([-dx, -dy, dz], axis=-1)
    normal_map = (normal_map + 1) * 127.5  # Normalisieren auf [0, 255]
    return normal_map.astype(np.uint8)

# 3. Roughness Map aus der Base Color Map generieren
def generate_roughness_map(base_color):
    # Vereinfachte Annahme: Dunklere Bereiche sind rauer
    grayscale = base_color.mean(axis=-1)  # In Graustufen konvertieren
    roughness_map = (255 - grayscale).astype(np.uint8)  # Invertieren
    return np.stack([roughness_map] * 3, axis=-1)  # In 3 Kanäle konvertieren

# 4. Occlusion Map aus der Base Color Map generieren
def generate_occlusion_map(base_color):
    # Vereinfachte Annahme: Dunklere Bereiche haben mehr Occlusion
    grayscale = base_color.mean(axis=-1)  # In Graustufen konvertieren
    occlusion_map = (255 - grayscale).astype(np.uint8)  # Invertieren
    return np.stack([occlusion_map] * 3, axis=-1)  # In 3 Kanäle konvertieren

# 5. Metallic Map (standardmäßig nicht-metallisch)
def generate_metallic_map(base_color):
    height, width, _ = base_color.shape
    return np.zeros((height, width, 3), dtype=np.uint8)  # Schwarz

# 6. Emission Map (standardmäßig keine Emission)
def generate_emission_map(base_color):
    height, width, _ = base_color.shape
    return np.zeros((height, width, 3), dtype=np.uint8)  # Schwarz

# 7. Alpha Map (standardmäßig undurchsichtig)
def generate_alpha_map(base_color):
    height, width, _ = base_color.shape
    return np.full((height, width, 3), 255, dtype=np.uint8)  # Weiß

# 8. GLTF 2.0-Datei erstellen
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

# 9. Hauptfunktion mit Tkinter
def main():
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

    # PBR-Karten generieren
    normal_map = generate_normal_map(base_texture)
    occlusion_map = generate_occlusion_map(base_texture)
    metallic_map = generate_metallic_map(base_texture)
    roughness_map = generate_roughness_map(base_texture)
    emission_map = generate_emission_map(base_texture)
    alpha_map = generate_alpha_map(base_texture)

    # GLTF-Datei erstellen
    create_gltf(base_texture, normal_map, occlusion_map, metallic_map, roughness_map, emission_map, alpha_map, output_path)

if __name__ == "__main__":
    main()