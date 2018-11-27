from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import re
import os
import errno

DATASET_BASE_DIR = "./dataset/"
SPRITE_BASE_DIR = DATASET_BASE_DIR + 'sprites/'


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


def get_sprite_image_urls(pokemon):
    raw_html = simple_get("https://pokemondb.net/sprites/{}".format(pokemon))
    soup = BeautifulSoup(raw_html, 'html.parser')
    img_tags = soup.find_all('span', class_=re.compile('img-fixed'))
    exclude = re.compile('.*(red|blue|silver|shiny).*', re.IGNORECASE)

    urls = []

    for tag in img_tags:
        if exclude.match(tag.attrs['data-alt']) is None:
            urls.append(tag.attrs['data-src'])
    return urls

def download_sprite_images(pokemon):
    os.makedirs(SPRITE_BASE_DIR + pokemon, exist_ok=True)

    urls = get_sprite_image_urls(pokemon)

    for i in range(len(urls)):
        print("Downloading image {}/{} for {}\n".format(i, len(urls), pokemon))
        filename = SPRITE_BASE_DIR + pokemon + '/{}.jpg'.format(i)
        image = get(urls[i], stream=True)
        with open(filename, "wb") as f:
            f.write(image.content)


def get_card_image_urls(pokemon):
    pass

if __name__ == "__main__":


    download_sprite_images("pikachu")




