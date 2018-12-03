import cv2 as cv
import os



def create_template(image):
    return cv.Canny(image, 100, 100)


def match_template(image, template):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    m = cv.matchTemplate(gray_image, gray_template, cv.TM_SQDIFF_NORMED)

    if m < 0.06:
        return True
    else:
        return False

def match_feature(image, template, ratio):
    orb = cv.ORB_create()

    kp1, des1 = orb.detectAndCompute(image, None)
    kp2, des2 = orb.detectAndCompute(template, None)

    if len(kp1) == 0 or len(kp2) == 0:
        return False

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(des1, des2)

    if len(matches) / len(kp1) > ratio:
        return True
    else:
        return False

def match_gradient_feature(image, template, ratio):
    orb = cv.ORB_create()

    image_gradient = cv.Canny(image, 100, 100)
    template_gradient = cv.Canny(template, 100, 100)

    kp1, des1 = orb.detectAndCompute(image_gradient, None)
    kp2, des2 = orb.detectAndCompute(template_gradient, None)

    if len(kp1) == 0 or len(kp2) == 0:
        return False

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(des1, des2)

    if len(matches) / len(kp1) > ratio:
        return True
    else:
        return False

def find_frames(video_path, query_image):
    cap = cv.VideoCapture(video_path)
    fourcc = cv.VideoWriter_fourcc('X', 'V', 'I', 'D')
    fps = cap.get(cv.CAP_PROP_FPS)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    count = 0
    is_action = False

    clip_num = 0
    #out = cv.VideoWriter('clips/clip_{}.mkv'.format(clip_num), fourcc, 24, size, True)

    while True:

        success, image = cap.read()
        buffer = []
        if len(buffer) < 24:
            buffer.append(image)
        else:
            buffer.pop(0)
            buffer.append(image)


        if not success or count > 18000:
            break
        # if count > 200:
        #     out.release()
        #     break
        # else:
        #     out.write(image)

        if count % 1000 == 0:
            print("At frame {}".format(count))

        if is_action:
            out.write(image)


        matched = match_gradient_feature(image, query_image, 0.7)
        if matched and not is_action:
            print("writing video ", count)
            is_action = True
            post_buffer = 0
            out = cv.VideoWriter('clips/clip_{}.mkv'.format(clip_num), fourcc, fps, size)
            for f in buffer:
                out.write(f)

            out.write(image)

        elif is_action and not matched and post_buffer < 24:
            print("writing post buffer")
            out.write(image)
            post_buffer += 1

        elif is_action and not matched:
            print("stopping video ", count)
            out.release()
            clip_num += 1
            is_action = False





        count += 1

    cap.release()

if __name__ == "__main__":
    query = 'frames/frame{}.jpg'.format(17699)
    template = cv.imread(query)

    find_frames('pokemon.mp4', template)

    # start = 17702
    # query = 'frames/frame{}.jpg'.format(17699)
    #
    # template = cv.imread(query)
    # template_gray = cv.imread(query, cv.IMREAD_GRAYSCALE)
    # template_edge = cv.Canny(template, 100, 100)
    #
    #
    # for i in range(10):
    #     filename = 'frames/frame{}.jpg'.format(start + i)
    #
    #     image = cv.imread(filename)
    #     image_gray = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    #     edge = cv.Canny(image, 100, 100)
    #     print("results for image {}".format(start + i))
    #     print("Colour x Colour")
    #     print(cv.matchTemplate(image, template, cv.TM_SQDIFF_NORMED))
    #
    #
    #
    #
    #     print("Gray x Gray")
    #     print(cv.matchTemplate(image_gray, template_gray, cv.TM_SQDIFF_NORMED))
    #
    #     print("Gray x Edge")
    #     print(cv.matchTemplate(image_gray, template_edge, cv.TM_SQDIFF_NORMED))
    #
    #
    #
    #
    #     print("Edge x Gray")
    #     print(cv.matchTemplate(edge, template_gray, cv.TM_SQDIFF_NORMED))
    #
    #     print("Edge x Edge")
    #     print(cv.matchTemplate(edge, template_edge, cv.TM_SQDIFF_NORMED))
    #
    #     print('\n\n\n')
    #
    # print("END")