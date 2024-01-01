from modules import sift, orb, template_matching


if __name__ == "__main__":
    im_path_1 = "./input/facial_features/eyes/eyes_2.jpg"
    im_path_2 = "./input/models/park seo joon.jpg"
    
    """sift.perform_sift(
        im_path_1, im_path_2
    }"""

    template_matching.perform_template_matching(im_path_2, im_path_1)

