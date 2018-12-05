import cv2 as cv

# matches the image and the template with template matching
# uses normalized squared difference as the score and a threshold of < 0.06 difference
# for a match
def match_template(image, template):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    m = cv.matchTemplate(gray_image, gray_template, cv.TM_SQDIFF_NORMED)

    if m < 0.06:
        return True
    else:
        return False

# matches the image and the template with ORB features
# match between the images is true if the number of matched features divided by the total
# number of features in the image is greater than ratio
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

# match the gradients of the image and the template using ORB features
# match between the images is true if the number of matched features divided by the total
# number of features in the image is greater than ratio
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

# iterates over the frames of the video to find clips containing the query image
# saves the clips to the directory clips/
# to create the clip, 24 buffer frames are stored for right before the detection and after the detection
# these frames along with the detected matching frames are written to a video file
def find_frames(video_path, query_image):
    cap = cv.VideoCapture(video_path)
    fourcc = cv.VideoWriter_fourcc('X', 'V', 'I', 'D')
    fps = cap.get(cv.CAP_PROP_FPS)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    count = 0
    is_action = False
    clip_num = 0

    # main loop that goes over the video frames
    while True:
        success, image = cap.read()

        # stores pre buffers
        buffer = []
        if len(buffer) < 24:
            buffer.append(image)
        else:
            buffer.pop(0)
            buffer.append(image)

        # breaks the loop when there are no more frames
        if not success:
            break

        if count % 1000 == 0:
            print("At frame {}".format(count))

        if is_action:
            out.write(image)

        # a match is only made when both the template and the features agree on a match
        matched = match_feature(image, template, 0.7) and match_template(image, template)
        if matched and not is_action:
            # begins writing the video
            print("writing video ", count)
            is_action = True
            post_buffer = 0
            out = cv.VideoWriter('clips/clip_{}.mkv'.format(clip_num), fourcc, fps, size)
            for f in buffer:
                out.write(f)

            out.write(image)

        elif is_action and not matched and post_buffer < 24:
            # writes the post buffer frames
            print("writing post buffer")
            out.write(image)
            post_buffer += 1

        elif is_action and not matched:
            # ends the video
            print("stopping video ", count)
            out.release()
            clip_num += 1
            is_action = False

        count += 1
    cap.release()

# searches the video pokemon.mp4 for the frame 17699
# needs to run the scrip extract_frames.py first
if __name__ == "__main__":

    query = 'frames/frame{}.jpg'.format(17699)
    template = cv.imread(query)

    find_frames('pokemon.mp4', template)

    # this section experiements with different template matching scores, such as normalized cross correlation
    # and normalized squared difference

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
    #     print("Gray x Gray")
    #     print(cv.matchTemplate(image_gray, template_gray, cv.TM_SQDIFF_NORMED))
    #
    #     print("Gray x Edge")
    #     print(cv.matchTemplate(image_gray, template_edge, cv.TM_SQDIFF_NORMED))
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