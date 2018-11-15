from urllib.request import urlopen
from bs4 import BeautifulSoup
import cognitive_face as cf
import requests
from io import BytesIO
from PIL import Image
from time import sleep
import pandas as pd
import Config


cf.Key.set(Config.AZURE_API_KEY)

cf.BaseUrl.set(Config.FACE_API_URL)


def run():
    img_urls = get_img_urls(Config.BLOG_URL)
    rows = []
    for img_url in img_urls:
        if not img_url.endswith('.jpg'):
            continue
        data_list = detect_faces(img_url)
        for data in data_list:
            rows.append(data)
    df = pd.DataFrame(rows, columns=['face_id', 'anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'])
    print(df)
    df.to_csv(Config.WORK_SPACE + 'result.csv')


def get_img_urls(base_url):
    html = urlopen(base_url)
    soup = BeautifulSoup(html, 'html.parser')

    img_elms = soup.find_all('img', class_='hatena-fotolife')
    img_urls = []
    for elm in img_elms:
        src = elm['src']
        print(src)
        img_urls.append(src)
    return img_urls


def detect_faces(img_url):
    data_list = []
    faces = cf.face.detect(img_url, attributes='emotion')
    print(faces)

    def get_rectangle(face_dictionary):
        rect = face_dictionary['faceRectangle']
        left = rect['left']
        top = rect['top']
        bottom = left + rect['height']
        right = top + rect['width']
        return (left, top, bottom, right)

    def get_emotion(face_dictionary):
        emotion = face_dictionary['faceAttributes']['emotion']
        anger = emotion['anger']
        contempt = emotion['contempt']
        disgust = emotion['disgust']
        fear = emotion['fear']
        happiness = emotion['happiness']
        neutral = emotion['neutral']
        sadness = emotion['sadness']
        surprise = emotion['surprise']
        return [anger, contempt, disgust, fear, happiness, neutral, sadness, surprise]

    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    for face in faces:
        rectangle = get_rectangle(face)
        img_cropped = img.crop(rectangle)
        face_id = face['faceId']
        dl_path = Config.WORK_SPACE + 'img/' + face_id + '.jpg'
        img_cropped.save(dl_path)
        data_list.append([face_id] + get_emotion(face))

    sleep(5)
    return data_list


if __name__ == '__main__':
    run()
