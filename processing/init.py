from GaussDenoise import process_single_image
import matplotlib.pyplot as plt

result = process_single_image("image.png", k=0.08)

print(result["psnr"], result["ssim"], result["snr_processed"])



plt.subplot(1,2,1)
plt.imshow(result["original"], cmap="gray")
plt.title(f"Original\nSNR={result['snr_original']:.2f}")

plt.subplot(1,2,2)
plt.imshow(result["processed"], cmap="gray")
plt.title(f"Processed\nPSNR={result['psnr']:.2f}, SSIM={result['ssim']:.2f}")

plt.show()
