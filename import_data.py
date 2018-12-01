from requests import get
from requests import exceptions
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import cv2 as cv
import re
import os
import sys
import json

NAME_ID_MAP = json.load(open('./names.json', 'r'))

DATASET_BASE_DIR = "./dataset/"
BING_BASE_DIR = DATASET_BASE_DIR + "bing/"
SPRITE_BASE_DIR = DATASET_BASE_DIR + 'sprites/'

API_KEY = "de35b454146141e5a071c2ad84fc07a4"
MAX_RESULTS = 200
GROUP_SIZE = 50

URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])



def query_bing(query, group_size=GROUP_SIZE, max_result=MAX_RESULTS):
    # store the search term in a convenience variable then set the
    # headers and search parameters
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    params = {"q": query, "offset": 0, "count": group_size}

    # make the search
    print("[INFO] searching Bing API for '{}'".format(query))
    search = get(URL, headers=headers, params=params)
    search.raise_for_status()

    # grab the results from the search, including the total number of
    # estimated results returned by the Bing API
    results = search.json()
    estNumResults = min(results["totalEstimatedMatches"], max_result)
    print("[INFO] {} total results for '{}'".format(estNumResults,
                                                    query))

    # initialize the total number of images downloaded thus far
    total = 0

    # loop over the estimated number of results in `GROUP_SIZE` groups
    for offset in range(0, estNumResults, group_size):
        # update the search parameters using the current offset, then
        # make the request to fetch the results
        print("[INFO] making request for group {}-{} of {}...".format(
            offset, offset + group_size, estNumResults))
        params["offset"] = offset
        search = get(URL, headers=headers, params=params)
        search.raise_for_status()
        results = search.json()
        print("[INFO] saving images for group {}-{} of {}...".format(
            offset, offset + group_size, estNumResults))

        # loop over the results
        for v in results["value"]:
            # try to download the image
            try:
                # make a request to download the image
                print("[INFO] fetching: {}".format(v["contentUrl"]))
                r = get(v["contentUrl"], timeout=30)

                # build the path to the output image
                ext = v["contentUrl"][v["contentUrl"].rfind("."):]
                if ext == '.gif':
                    continue
                os.makedirs(BING_BASE_DIR + query, exist_ok=True)
                p = "{}{}/{}{}".format(BING_BASE_DIR, query,
                    str(total).zfill(8), ext)

                # write the image to disk
                with open(p, "wb") as f:
                    f.write(r.content)

            # catch any errors that would not unable us to download the
            # image
            except Exception as e:
                # check to see if our exception is in our list of
                # exceptions to check for
                if type(e) in EXCEPTIONS:
                    print("[INFO] skipping: {}".format(v["contentUrl"]))
                    continue

            # try to load the image from disk
            image = cv.imread(p)

            # if the image is `None` then we could not properly load the
            # image from disk (so it should be ignored)
            if image is None:
                print("[INFO] deleting: {}".format(p))
                os.remove(p)
                continue

            # update the counter
            total += 1


def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


def get_sprite_image_urls(pokemon, all=True):
    raw_html = simple_get("https://pokemondb.net/sprites/{}".format(pokemon))
    soup = BeautifulSoup(raw_html, 'html.parser')
    img_tags = soup.find_all('span', class_=re.compile('img-fixed'))
    image_urls = [tag.attrs['data-src'] for tag in img_tags]


    if all:
        exclude = re.compile('.*(red|blue|silver|shiny).*', re.I)
    else:
        exclude = re.compile('.*(red|blue|silver|shiny|back).*', re.I)

    urls = [u for u in image_urls if exclude.match(u) is None]
    return urls

def download_sprite_images(pokemon, all=True, mute=True):
    os.makedirs(SPRITE_BASE_DIR + pokemon, exist_ok=True)

    urls = get_sprite_image_urls(pokemon, all)

    for i in range(len(urls)):
        if not mute:
            print("Downloading image {}/{} for {}\n".format(i, len(urls), pokemon))
        filename = SPRITE_BASE_DIR + pokemon + '/{}.jpg'.format(i)
        image = get(urls[i], stream=True)
        with open(filename, "wb") as f:
            f.write(image.content)

    print("Sprite images downloaded to /dataset/sprites/{}\n".format(pokemon))


def get_card_image_urls(pokemon):
    pass

def create_name_id_mapping(filename):
    pass

if __name__ == "__main__":

    if len(sys.argv) > 2:
        MAX_RESULTS = sys.argv[2]

    if len(sys.argv) > 1:
        pokemon_name = sys.argv[1]
        download_sprite_images(pokemon_name, all=False, mute=False)

        query_bing(pokemon_name)
    else:

        for name in ["pikachu", "gengar"]:#list(NAME_ID_MAP.keys())[:3]:
            print("Downloading images for {}\n".format(name))
            query_bing(name, 50, 200)
            #download_sprite_images(name, all=False, mute=False)


