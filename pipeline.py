from MaskTheFace.mask_the_face_fun import *
from face_detection import face_detection_crop
from occlusion_blackbox.sunglasses import sunglass_occlusion
from occlusion_blackbox.black_box import percent_ten_blackbox_single
from loguru import logger
import argparse
# 3 different function for 3 different task 
def black(input_dir,output_dir_black,percent): # isme percent mein hi  pass kardenge 10 ya 20 ya 40
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                output_path = os.path.join(output_dir_black, os.path.relpath(img_path, input_dir))

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image=cv2.imread(img_path)
                if image is None:
                    logger.error(f"Skipping invalid image: {img_path}")
                    continue
                croped,x,y,w,h=face_detection_crop(image)
                black_image=percent_ten_blackbox_single(croped,x,y,w,h,percent)
                cv2.imwrite(output_path, black_image)
                logger.info(f"Processed and saved: {output_path}")

def glasses(input_dir,output_dir_glasses,sunglass_image_path):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                output_path = os.path.join(output_dir_glasses, os.path.relpath(img_path, input_dir))

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image=cv2.imread(img_path)
                if image is None:
                    logger.error(f"Skipping invalid image: {img_path}")
                    continue
                croped,x,y,w,h=face_detection_crop(image)
                glasses_image=sunglass_occlusion(croped,x,y,w,h,sunglass_image_path)
                cv2.imwrite(output_path, glasses_image)
                logger.info(f"Processed and saved: {output_path}")
def mask(input_dir,output_dir_mask): # isme bcs phele se likha code change karna hai tu uska function mein thodi samasyaa hai
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="to make occluded dataset "
    )

    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="input directory of images",
    )
    parser.add_argument(
        "--output_blackbox_path",
        type=str,
        default="",
        help="output black box  directory of images",
    )
    parser.add_argument(
        "--output_glasses_path",
        type=str,
        default="",
        help="output glasses directory of images",
    )
    parser.add_argument(
        "--glass_image_path",
        type=str,
        default="",
        help="input of glass  images",
    )
    parser.add_argument(
        "--percent",
        type=str,
        default="",
        help="percent black box occlusion ",
    )
    args = parser.parse_args()
    black(args.path,args.output_blackbox_path,args.percent)
    glasses(args.path,args.output_glasses_path,args.glass_image_path)

        
        






