import cv2
import numpy as np


x = cv2.imread('original_img.PNG', cv2.IMREAD_UNCHANGED)
rgb = x[:, :, :-1] / 255
alpha = x[:, :, -1]
mask = np.where(alpha > 50, 1, 0).astype(np.uint8)

strip_img, pix_map = create_strip(rgb, mask)

reconstructed_image = reconstruct_strip(rgb, strip_img, pix_map, 80)
cv2.imwrite("strip.jpg", np.round(255*strip_img))
np.save('pixel_map', pix_map, allow_pickle=True)
cv2.imwrite("reconstructed.jpg", np.round(255*reconstructed_image))
