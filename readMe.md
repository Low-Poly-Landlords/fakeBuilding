The files in this project serve 2 purposes:
1. To create floorplans, a lidar scan, etc. for a completely fictional hotel. This data is used to test other areas of the project, when the rest of the code isn't working all the way or isn't finished.
2. To take a .mcap file from a scan on our device and create a 3d model of a room and a 3d floor plan of said room.

# To sucessfully run the programs:

## To run the fake hotel programs:
1. Run fakeHotelLayout.py
    - This program generates a .json file that contains the whole layout of the fake hotel.
2. Run fakeHotelViewer.py
    - This allows you to view a 3d model of the hotel, and is not strictly necessary but is nice to get an idea of what you're working towards.
3. Run fakeHotelFloorPlanMaker.py
    - This creates a .pdf file that contains the floor plans of the hotel.
4. Run fakeHotelScanSim.py
    - This will take a while. It simulates a person walking through the hotel and using our device to gather a point cloud to make the 3d model with.
5. Run fakeHotelScanToModel.py
    - This will take a while, but less time than the previous step. It uses the generated .mcap file to create a 3d model of the hotel.

## To run the real scan programs:
1. Run cameraAlignment.py
    - This program uses camera data to generate a good angle of pitch for the camera. This helps line up the image with the lidar data, but I guess it's not necessary
    - Once you've ran this, edit the code of colorPipeline.py and rigidBlockPipelin.py to have this angle.
2. Run either:
    - rigidBlockPipeline
        - This generates a "snapped to a grid" version of the room with the most clear images projected onto the wall. (how the images are projected will be changed in the future)
        - This also generates a .dxf file containing a 2d floor plan, with material properties in the walls that can later be used to simulate attenuation
    - colorPipeline
        - This generates a voxelized version of the room, where the voxels are painted the color of the "most commmon" color that appears in that spot in the camera data.
        - This DOES NOT generate a 2d floor plan as of right now
