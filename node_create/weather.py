import requests
from urllib import parse
import json
from pyproj import Proj, transform
from flask import Blueprint, jsonify
import math
import datetime

weather_blueprint = Blueprint('weather', __name__)

global_base_time_str = ''
global_base_date_str = ''

@weather_blueprint.route('/get_weather_data', methods=['GET'])
def get_weather_data():

    def checkTime():
        # 현재 시간 얻기
        current_time = datetime.datetime.now()

        # 현재 시간의 분 확인
        current_minute = current_time.minute

        # 분이 40분 이상이면 다음 시간대로 설정
        if current_minute >= 40:
            base_time = current_time
        else:
            base_time = current_time - datetime.timedelta(hours=1)
        # 'base_time' 설정 (예: 18:00 또는 19:00)
        base_time = base_time.replace(minute=0, second=0, microsecond=0)

        global global_base_time_str
        global global_base_date_str
        # 'base_time'을 문자열로 변환 (예: '1800' 또는 '1900')
        base_time_str = base_time.strftime('%H%M')
        global_base_time_str = base_time_str

        year_and_date = current_time.strftime("%Y-%m-%d")
        global_base_date_str = year_and_date.replace("-", "")
        print('현재 시간:', current_time)
        print('base_time:', base_time_str)

    checkTime()

    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'
    key = 'vE5pv9fzPca1B7EEvgAkv48NgFzgTHe2HePqIQg0rRxyklrVJAKAxBjgmjXHRL+3ErTVNH4PA8RbszGaNolAtw=='
    serviceKeyDecoded = parse.unquote(key, 'UTF-8')

    NX = 80  # X축 격자점 수
    NY = 75  # Y축 격자점 수

    params ={'serviceKey': key, 'pageNo': '1', 'numOfRows': '1000', 'dataType': 'JSON', 'base_date': global_base_date_str,
             'base_time': global_base_time_str, 'nx': NX, 'ny': NY }

    response = requests.get(url, params=params)

    # 응답 JSON 파싱
    data = json.loads(response.content)
    filtered_data = data['response']['body']['items']['item']
    
    wind_data = []
    for item in filtered_data:
        if item['category'] == 'VVV' or item['category'] == 'WSD':
            wind_data.append(item)
    
    print(wind_data)
    return jsonify(wind_data)

    # with open('weather_data.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(data, json_file, ensure_ascii=False, indent=4)
    # print('데이터가 "weather_data.json" 파일로 저장되었습니다.')


Re = 6371.00877     ##  지도반경
grid = 5.0          ##  격자간격 (km)
slat1 = 30.0        ##  표준위도 1
slat2 = 60.0        ##  표준위도 2
olon = 126.0        ##  기준점 경도
olat = 38.0         ##  기준점 위도
xo = 210 / grid     ##  기준점 X좌표
yo = 675 / grid     ##  기준점 Y좌표
first = 0

if first == 0 :
    PI = math.asin(1.0) * 2.0
    DEGRAD = PI/ 180.0
    RADDEG = 180.0 / PI


    re = Re / grid
    slat1 = slat1 * DEGRAD
    slat2 = slat2 * DEGRAD
    olon = olon * DEGRAD
    olat = olat * DEGRAD

    sn = math.tan(PI * 0.25 + slat2 * 0.5) / math.tan(PI * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(PI * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(PI * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)
    first = 1

def mapToGrid(lon, lat, code = 0 ):
    ra = math.tan(PI * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / pow(ra, sn)
    theta = lon * DEGRAD - olon
    if theta > PI :
        theta -= 2.0 * PI
    if theta < -PI :
        theta += 2.0 * PI
    theta *= sn
    x = (ra * math.sin(theta)) + xo
    y = (ro - ra * math.cos(theta)) + yo
    x = int(x + 1.5)
    y = int(y + 1.5)
    return x, y

def gridToMap(x, y, code = 1):
    x = x - 1
    y = y - 1
    xn = x - xo
    yn = ro - y + yo
    ra = math.sqrt(xn * xn + yn * yn)
    if sn < 0.0 :
        ra = -ra
    alat = math.pow((re * sf / ra), (1.0 / sn))
    alat = 2.0 * math.atan(alat) - PI * 0.5
    if math.fabs(xn) <= 0.0 :
        theta = 0.0
    else :
        if math.fabs(yn) <= 0.0 :
            theta = PI * 0.5
            if xn < 0.0 :
                theta = -theta
        else :
            theta = math.atan2(xn, yn)
    alon = theta / sn + olon
    lat = alat * RADDEG
    lon = alon * RADDEG

    return lon, lat

print(mapToGrid(128.0985525867203, 35.15399396813463))
print(gridToMap(80, 75))
