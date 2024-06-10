## Panorama stitching from Scratch

## Algorithm Steps

For each subsequent image, perform the following steps:

1. **Use SIFT feature matching** to find the transformation matrix between the current panorama and the new image.
2. **Define the corner points** of both images.
3. **Warp the corner points** of the new image using the transformation matrix.
4. **Calculate the minimum and maximum coordinates** of the combined corners.
5. **Shift the transformation matrix** to align the images properly.
6. **Calculate the canvas size** for the panorama.
7. **Warp both images** to the panorama canvas.
8. **Blend the warped images** to create the panorama.

![panorama](https://github.com/atreyabhat/RBE-ComputerVision/assets/39030188/bac9b73e-f66a-4833-a678-5c6f40c02a17)
![custom_pano](https://github.com/atreyabhat/RBE-ComputerVision/assets/39030188/b85611d5-404d-493f-bcc0-4cbaa3e1cd3b)
![boston_pano](https://github.com/atreyabhat/RBE-ComputerVision/assets/39030188/d2f88250-662c-41b2-ae4c-daa85619c251)
