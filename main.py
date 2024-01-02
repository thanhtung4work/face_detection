from modules import sift, orb, template_matching, harris_corner, hog


if __name__ == "__main__":
    im_path_1 = "./input/facial_features/eyes/eyes_1.jpg"
    im_path_2 = "./input/models/keanu reaves.jpg"
    
    if 0: sift.perform_sift(
        im_path_1, im_path_2
    )

    if 0: orb.perform_orb(
        im_path_1, im_path_2
    )

    if 0: template_matching.perform_template_matching(im_path_2, im_path_1)

    if 0: harris_corner.perform_harris_corner(im_path_1, im_path_2)

    if 1: hog.perform_hog(im_path_2)

