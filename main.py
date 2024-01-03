import glob

from modules import sift, orb, template_matching, harris_corner, hog, haar_cascades, contour_comparison


if __name__ == "__main__":
    im_path_1 = "./input/facial_features/faces/face_1.jpg"
    im_path_2 = "./input/models/person.png"
    
    if 0: sift.perform_sift(
        im_path_2, im_path_1
    )

    if 0: orb.perform_orb(
        im_path_2, im_path_1
    )

    if 0: template_matching.perform_template_matching(im_path_2, im_path_1)

    if 0: harris_corner.perform_harris_corner(im_path_2, im_path_2)

    if 0: hog.perform_hog(im_path_2)

    if 1: 
        haar_cascades.perform_cascades(im_path_2)
        haar_cascades.perform_cascades(im_path_2, hierarchical=False)

    if 0: contour_comparison.perform_cnt_comparison(
        "./input/models/person.png", 
        glob.glob("./input/models/*")
    )