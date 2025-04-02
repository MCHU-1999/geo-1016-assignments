import pymeshlab
import time

POISSON_FOLDER = "./A3_Reconstruction_Data/surface_reconstruction/poisson_surface_reconstruction"
POISSON_FILES = [
    "Poisson_01_polyhedron.ply",
    "Poisson_02_SyntheticBuilding.ply",
    "Poisson_03_LongBuilding.ply",
    "Poisson_04_foampack.ply"
]

ms = pymeshlab.MeshSet()

if __name__ == "__main__":


    for file_name in POISSON_FILES:
        input_file = f"{POISSON_FOLDER}/{file_name}"
        
        print(f"Processing: {input_file}")
        
        # Load the point cloud
        ms.load_new_mesh(f"{POISSON_FOLDER}/{POISSON_FILES[0]}")

        # Start timing
        start_time = time.time()
    
        ms.generate_surface_reconstruction_screened_poisson(
            # depth=10, 
            # fulldepth=10
        )
        print("done")
                
        # Stop timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Reconstruction completed in {elapsed_time:.2f} seconds.")
