#!/bin/zsh
## 🏛️ 역사 및 지리
python scrape_wiki_custom.py --source-type category --source-value "한국의_역사" --max-articles 30 --output history_korea.jsonl
python scrape_wiki_custom.py --source-type category --source-value "세계의_역사" --max-articles 30 --output history_world.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한민국의_지리" --max-articles 30 --output geography_korea.jsonl
python scrape_wiki_custom.py --source-type category --source-value "아시아의_지리" --max-articles 30 --output geography_asia.jsonl
python scrape_wiki_custom.py --source-type category --source-value "한국_전쟁" --max-articles 30 --output history_korean_war.jsonl
python scrape_wiki_custom.py --source-type category --source-value "일제강점기" --max-articles 30 --output history_colonial_era.jsonl
python scrape_wiki_custom.py --source-type category --source-value "조선" --max-articles 30 --output history_joseon.jsonl
python scrape_wiki_custom.py --source-type category --source-value "고려" --max-articles 30 --output history_goryeo.jsonl
python scrape_wiki_custom.py --source-type category --source-value "서울의_역사" --max-articles 30 --output history_seoul.jsonl

## 🔬 과학 및 기술
python scrape_wiki_custom.py --source-type category --source-value "과학" --max-articles 25 --output science_general.jsonl
python scrape_wiki_custom.py --source-type category --source-value "물리학" --max-articles 25 --output science_physics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "화학" --max-articles 25 --output science_chemistry.jsonl
python scrape_wiki_custom.py --source-type category --source-value "생물학" --max-articles 25 --output science_biology.jsonl
python scrape_wiki_custom.py --source-type category --source-value "지구과학" --max-articles 25 --output science_earth.jsonl
python scrape_wiki_custom.py --source-type category --source-value "컴퓨터_과학" --max-articles 25 --output tech_computer_science.jsonl
python scrape_wiki_custom.py --source-type category --source-value "인공지능" --max-articles 25 --output tech_ai.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한민국의_기술" --max-articles 25 --output tech_korea.jsonl
python scrape_wiki_custom.py --source-type category --source-value "우주_기술" --max-articles 25 --output tech_space.jsonl
python scrape_wiki_custom.py --source-type category --source-value "의학" --max-articles 25 --output science_medicine.jsonl

## 🎨 문화 및 예술
python scrape_wiki_custom.py --source-type category --source-value "한국_문화" --max-articles 20 --output culture_korea.jsonl
python scrape_wiki_custom.py --source-type category --source-value "미술" --max-articles 20 --output art_general.jsonl
python scrape_wiki_custom.py --source-type category --source-value "음악" --max-articles 20 --output art_music.jsonl
python scrape_wiki_custom.py --source-type category --source-value "영화" --max-articles 20 --output art_film.jsonl
python scrape_wiki_custom.py --source-type category --source-value "문학" --max-articles 20 --output art_literature.jsonl
python scrape_wiki_custom.py --source-type category --source-value "건축" --max-articles 20 --output art_architecture.jsonl
python scrape_wiki_custom.py --source-type category --source-value "K-pop" --max-articles 20 --output culture_kpop.jsonl
python scrape_wiki_custom.py --source-type category --source-value "한국_요리" --max-articles 20 --output culture_korean_food.jsonl
python scrape_wiki_custom.py --source-type category --source-value "종교" --max-articles 20 --output culture_religion.jsonl
python scrape_wiki_custom.py --source-type category --source-value "신화" --max-articles 20 --output culture_mythology.jsonl

## 🏛️ 정치 및 경제
python scrape_wiki_custom.py --source-type search --source-value "대한민국 정치" --max-articles 15 --output politics_korea.jsonl
python scrape_wiki_custom.py --source-type category --source-value "경제학" --max-articles 15 --output economy_economics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한민국의_경제" --max-articles 15 --output economy_korea.jsonl
python scrape_wiki_custom.py --source-type category --source-value "법" --max-articles 15 --output society_law.jsonl
python scrape_wiki_custom.py --source-type category --source-value "국제_관계" --max-articles 15 --output politics_international.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한민국의_정당" --max-articles 15 --output politics_korean_parties.jsonl
python scrape_wiki_custom.py --source-type category --source-value "사회학" --max-articles 15 --output society_sociology.jsonl
python scrape_wiki_custom.py --source-type category --source-value "금융" --max-articles 15 --output economy_finance.jsonl

## 🌳 인물 및 사회
python scrape_wiki_custom.py --source-type category --source-value "한국의_인물" --max-articles 20 --output people_korea.jsonl
python scrape_wiki_custom.py --source-type category --source-value "철학" --max-articles 20 --output society_philosophy.jsonl
python scrape_wiki_custom.py --source-type category --source-value "교육" --max-articles 20 --output society_education.jsonl
python scrape_wiki_custom.py --source-type category --source-value "스포츠" --max-articles 20 --output society_sports.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한민국의_사회" --max-articles 20 --output society_korea.jsonl
python scrape_wiki_custom.py --source-type category --source-value "언어학" --max-articles 20 --output society_linguistics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "심리학" --max-articles 20 --output society_psychology.jsonl
python scrape_wiki_custom.py --source-type category --source-value "환경" --max-articles 20 --output society_environment.jsonl
python scrape_wiki_custom.py --source-type category --source-value "축제" --max-articles 20 --output society_festivals.jsonl

## 🐾 생물 및 자연
python scrape_wiki_custom.py --source-type category --source-value "동물" --max-articles 25 --output nature_animals.jsonl
python scrape_wiki_custom.py --source-type category --source-value "식물" --max-articles 25 --output nature_plants.jsonl
python scrape_wiki_custom.py --source-type category --source-value "한반도의_생물" --max-articles 25 --output nature_korean_peninsula.jsonl
python scrape_wiki_custom.py --source-type category --source-value "지질학" --max-articles 25 --output nature_geology.jsonl
python scrape_wiki_custom.py --source-type category --source-value "기후" --max-articles 25 --output nature_climate.jsonl


## 🏛️ 역사 및 군사 (세분화)
python scrape_wiki_custom.py --source-type category --source-value "고조선" --max-articles 30 --output history_gojoseon.jsonl
python scrape_wiki_custom.py --source-type category --source-value "삼국_시대" --max-articles 30 --output history_three_kingdoms.jsonl
python scrape_wiki_custom.py --source-type category --source-value "고구려" --max-articles 30 --output history_goguryeo.jsonl
python scrape_wiki_custom.py --source-type category --source-value "백제" --max-articles 30 --output history_baekje.jsonl
python scrape_wiki_custom.py --source-type category --source-value "신라" --max-articles 30 --output history_silla.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한제국" --max-articles 30 --output history_korean_empire.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한민국_임시정부" --max-articles 30 --output history_provisional_gov.jsonl
python scrape_wiki_custom.py --source-type category --source-value "군사사" --max-articles 30 --output history_military.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한민국 국군" --max-articles 30 --output military_rok.jsonl
python scrape_wiki_custom.py --source-type category --source-value "고고학" --max-articles 30 --output history_archaeology.jsonl
python scrape_wiki_custom.py --source-type category --source-value "한국의_독립운동" --max-articles 30 --output history_independence_movement.jsonl

## 🔬 과학 및 기술 (세분화)
python scrape_wiki_custom.py --source-type category --source-value "수학" --max-articles 25 --output science_mathematics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "천문학" --max-articles 25 --output science_astronomy.jsonl
python scrape_wiki_custom.py --source-type category --source-value "유전학" --max-articles 25 --output science_genetics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "뇌과학" --max-articles 25 --output science_neuroscience.jsonl
python scrape_wiki_custom.py --source-type category --source-value "로봇공학" --max-articles 25 --output tech_robotics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "소프트웨어_공학" --max-articles 25 --output tech_software_engineering.jsonl
python scrape_wiki_custom.py --source-type category --source-value "정보_이론" --max-articles 25 --output tech_information_theory.jsonl
python scrape_wiki_custom.py --source-type category --source-value "암호학" --max-articles 25 --output tech_cryptography.jsonl
python scrape_wiki_custom.py --source-type category --source-value "생명공학" --max-articles 25 --output tech_biotechnology.jsonl
python scrape_wiki_custom.py --source-type category --source-value "나노기술" --max-articles 25 --output tech_nanotechnology.jsonl
python scrape_wiki_custom.py --source-type category --source-value "통계학" --max-articles 25 --output science_statistics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "양자역학" --max-articles 25 --output science_quantum_mechanics.jsonl

## 🎨 문화, 예술 및 스포츠 (세분화)
python scrape_wiki_custom.py --source-type category --source-value "한국의_미술" --max-articles 20 --output art_korean.jsonl
python scrape_wiki_custom.py --source-type category --source-value "한국의_음악" --max-articles 20 --output art_korean_music.jsonl
python scrape_wiki_custom.py --source-type category --source-value "한국의_영화" --max-articles 20 --output art_korean_film.jsonl
python scrape_wiki_custom.py --source-type category --source-value "한국의_문학" --max-articles 20 --output art_korean_literature.jsonl
python scrape_wiki_custom.py --source-type category --source-value "한국의_건축" --max-articles 20 --output art_korean_architecture.jsonl
python scrape_wiki_custom.py --source-type category --source-value "사진술" --max-articles 20 --output art_photography.jsonl
python scrape_wiki_custom.py --source-type category --source-value "디자인" --max-articles 20 --output art_design.jsonl
python scrape_wiki_custom.py --source-type category --source-value "만화" --max-articles 20 --output culture_comics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "애니메이션" --max-articles 20 --output culture_animation.jsonl
python scrape_wiki_custom.py --source-type category --source-value "무용" --max-articles 20 --output art_dance.jsonl
python scrape_wiki_custom.py --source-type category --source-value "공예" --max-articles 20 --output art_crafts.jsonl
python scrape_wiki_custom.py --source-type category --source-value "드라마" --max-articles 20 --output culture_drama.jsonl
python scrape_wiki_custom.py --source-type category --source-value "축구" --max-articles 20 --output sports_football.jsonl
python scrape_wiki_custom.py --source-type category --source-value "야구" --max-articles 20 --output sports_baseball.jsonl
python scrape_wiki_custom.py --source-type category --source-value "농구" --max-articles 20 --output sports_basketball.jsonl
python scrape_wiki_custom.py --source-type category --source-value "올림픽" --max-articles 20 --output sports_olympics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "e스포츠" --max-articles 20 --output sports_esports.jsonl
python scrape_wiki_custom.py --source-type category --source-value "한국의_무형문화재" --max-articles 20 --output culture_intangible_heritage.jsonl

## 🏛️ 정치, 경제 및 법률 (세분화)
python scrape_wiki_custom.py --source-type category --source-value "대한민국의_헌법" --max-articles 15 --output law_korean_constitution.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한민국의_대통령" --max-articles 15 --output politics_korean_presidents.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한민국_국회" --max-articles 15 --output politics_national_assembly.jsonl
python scrape_wiki_custom.py --source-type category --source-value "국제법" --max-articles 15 --output law_international.jsonl
python scrape_wiki_custom.py --source-type category --source-value "선거" --max-articles 15 --output politics_elections.jsonl
python scrape_wiki_custom.py --source-type category --source-value "외교" --max-articles 15 --output politics_diplomacy.jsonl
python scrape_wiki_custom.py --source-type category --source-value "무역" --max-articles 15 --output economy_trade.jsonl
python scrape_wiki_custom.py --source-type category --source-value "마케팅" --max-articles 15 --output economy_marketing.jsonl
python scrape_wiki_custom.py --source-type category --source-value "재무관리" --max-articles 15 --output economy_financial_management.jsonl
python scrape_wiki_custom.py --source-type category --source-value "세금" --max-articles 15 --output economy_tax.jsonl
python scrape_wiki_custom.py --source-type category --source-value "대한민국의_기업" --max-articles 15 --output economy_korean_companies.jsonl
python scrape_wiki_custom.py --source-type category --source-value "노동" --max-articles 15 --output economy_labor.jsonl

## 🌳 사회 및 인물 (세분화)
python scrape_wiki_custom.py --source-type category --source-value "언론" --max-articles 20 --output society_media.jsonl
python scrape_wiki_custom.py --source-type category --source-value "여성주의" --max-articles 20 --output society_feminism.jsonl
python scrape_wiki_custom.py --source-type category --source-value "인권" --max-articles 20 --output society_human_rights.jsonl
python scrape_wiki_custom.py --source-type category --source-value "도시" --max-articles 20 --output society_cities.jsonl
python scrape_wiki_custom.py --source-type category --source-value "관광" --max-articles 20 --output society_tourism.jsonl
python scrape_wiki_custom.py --source-type category --source-value "교통" --max-articles 20 --output society_transportation.jsonl
python scrape_wiki_custom.py --source-type category --source-value "철학자" --max-articles 20 --output people_philosophers.jsonl
python scrape_wiki_custom.py --source-type category --source-value "과학자" --max-articles 20 --output people_scientists.jsonl
python scrape_wiki_custom.py --source-type category --source-value "예술가" --max-articles 20 --output people_artists.jsonl
python scrape_wiki_custom.py --source-type category --source-value "정치인" --max-articles 20 --output people_politicians.jsonl
python scrape_wiki_custom.py --source-type category --source-value "기업인" --max-articles 20 --output people_entrepreneurs.jsonl
python scrape_wiki_custom.py --source-type category --source-value "한국의_성씨" --max-articles 20 --output culture_korean_surnames.jsonl
python scrape_wiki_custom.py --source-type category --source-value "민속" --max-articles 20 --output culture_folklore.jsonl

## 🐾 자연 및 환경 (세분화)
python scrape_wiki_custom.py --source-type category --source-value "포유류" --max-articles 25 --output nature_mammals.jsonl
python scrape_wiki_custom.py --source-type category --source-value "조류" --max-articles 25 --output nature_birds.jsonl
python scrape_wiki_custom.py --source-type category --source-value "어류" --max-articles 25 --output nature_fish.jsonl
python scrape_wiki_custom.py --source-type category --source-value "파충류" --max-articles 25 --output nature_reptiles.jsonl
python scrape_wiki_custom.py --source-type category --source-value "곤충" --max-articles 25 --output nature_insects.jsonl
python scrape_wiki_custom.py --source-type category --source-value "버섯" --max-articles 25 --output nature_mushrooms.jsonl
python scrape_wiki_custom.py --source-type category --source-value "해양생물학" --max-articles 25 --output science_marine_biology.jsonl
python scrape_wiki_custom.py --source-type category --source-value "생태학" --max-articles 25 --output science_ecology.jsonl
python scrape_wiki_custom.py --source-type category --source-value "기상학" --max-articles 25 --output science_meteorology.jsonl
python scrape_wiki_custom.py --source-type category --source-value "화산" --max-articles 25 --output nature_volcanoes.jsonl
python scrape_wiki_custom.py --source-type category --source-value "강" --max-articles 25 --output nature_rivers.jsonl
python scrape_wiki_custom.py --source-type category --source-value "산" --max-articles 25 --output nature_mountains.jsonl
python scrape_wiki_custom.py --source-type category --source-value "멸종" --max-articles 25 --output nature_extinction.jsonl

## 📚 철학 및 사상
python scrape_wiki_custom.py --source-type category --source-value "동양_철학" --max-articles 20 --output philosophy_eastern.jsonl
python scrape_wiki_custom.py --source-type category --source-value "서양_철학" --max-articles 20 --output philosophy_western.jsonl
python scrape_wiki_custom.py --source-type category --source-value "윤리학" --max-articles 20 --output philosophy_ethics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "논리학" --max-articles 20 --output philosophy_logic.jsonl
python scrape_wiki_custom.py --source-type category --source-value "미학" --max-articles 20 --output philosophy_aesthetics.jsonl
python scrape_wiki_custom.py --source-type category --source-value "정치철학" --max-articles 20 --output philosophy_political.jsonl
python scrape_wiki_custom.py --source-type category --source-value "불교" --max-articles 20 --output religion_buddhism.jsonl
python scrape_wiki_custom.py --source-type category --source-value "기독교" --max-articles 20 --output religion_christianity.jsonl
python scrape_wiki_custom.py --source-type category --source-value "이슬람교" --max-articles 20 --output religion_islam.jsonl
python scrape_wiki_custom.py --source-type category --source-value "유교" --max-articles 20 --output religion_confucianism.jsonl
