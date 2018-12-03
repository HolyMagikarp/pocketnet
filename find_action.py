import cv2 as cv
import os



def create_template(image):
    return cv.Canny(image, 100, 100)


def match_template(image, template):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    m = cv.matchTemplate(gray_image, gray_template, cv.TM_SQDIFF_NORMED)

    if m < 0.08:
        return True
    else:
        return False

def find_frames(video_path, query_image):
    cap = cv.VideoCapture(video_path)
    fourcc = cv.VideoWriter_fourcc('M','P','4','2')
    fps = cap.get(cv.CAP_PROP_FPS)


    count = 0
    is_action = False

    clip_num = 0
    while True:



        success, image = cap.read()
        out = cv.VideoWriter('clips/clip_{}.avi'.format(clip_num), fourcc, 24, image.shape[:2])

        # if count < 17600:
        #     continue
        if count > 1000:
            out.release()
            break
        else:
            out.write(image)

        if match_template(image, query_image):
            print("writing video ", count)
            is_action = True
            out = cv.VideoWriter('clips/clip_{}.avi'.format(clip_num), fourcc, fps, image.shape[:2])
        elif is_action:
            print("stopping video ", count)
            out.release()
            clip_num += 1

        if is_action:
            out.write(image)

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