# DCP - Bitska analiza

## Uputstvo za pokretanje na Linux sistemu

## 1. Zavisnosti.

Za izgradnju i pokretanje projekta potrebno je:
- C++ kompajler sa podrškom za C++17
- CMake verzije 3.16 ili novije
- OpenCV 4.x

## 2. Izgradnja i pokretanje na Linux sistemu.

1. Instalacija potrebnih paketa:
```
sudo apt update
sudo apt install cmake g++ libopencv-dev
```
2. Prelazak u direktorijum u kojem se nalazi projekat:
```
cd bit_analysis
```
3. Konfiguracija projekta i njegovo kompajliranje:
```
cmake -S . -B build
cmake --build build --config Release
```
4. Pokretanje programa:
```
./build/HazeRemoval_BitAnalysis dataset/
```
Programu se kao prvi argument komandne linije prosleđuje putanja do direktorijuma koji sadrži skup ulaznih slika. U ovom primeru se koristi direktorijum dataset.

Po potrebi, moguće je prilikom pokretanja programa proslediti dodatne argumente, za izbor slike koja će se koristiti kao vizuelni prikaz rezultata, kao i za izbor vrednosti broja razlomljenih bitova F, nad kojima se analiza izvršava:
```
./build/HazeRemoval_BitAnalysis dataset/ --image "naziv_slike.png" --frac-bits 7,8,9,10,11
```

Ukoliko se dodatni argumenti izostave, koriste se podrazumevane vrednosti: slika 'SGP_Google_345.png' za vizuelni prikaz rezultata, kao i vrednosti broja razlomljenih bitova F = 6, 8, 10, 12.

## 3. Izlazni rezultati

Tokom izvršavanja programa generišu se sledeći fajlovi:
- bit_analysis_global.csv - zbirna analiza opsega vrednosti međurezultata DCP algoritma.
- fixed_point_image_quality.csv - rezultati poređenja kvaliteta konačnih rekonstruisanih slika za svaku ulaznu sliku i svaku testiranu vrednost F.
- fixed_point_image_quality_summary.csv - zbirni rezultati poređenja kvaliteta konačnih slika za svaku testiranu vrednost F.

Za odabranu sliku dodatno se čuvaju:
- reference_double.png - rezultat dobijen korišćenjem referentne implementacije sa pokretnim zarezom.
- fixed_F6.png, fixed_F8.png itd. - rezultati dobijeni simulacijom formata fiksnog zareza za odgovarajuće vrednosti

Dobijeni rezultati čuvaju se u radnom direktorijumu iz kojeg je program pokrenut.
