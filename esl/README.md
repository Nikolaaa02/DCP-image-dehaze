# DCP - Izvršna specifikacija algoritma Dark Channel Prior

## Uputstvo za pokretanje na Linux sistemu

## 1. Zavisnosti

Za izgradnju i pokretanje projekta potrebno je:
- C++ kompajler sa podrškom za C++17
- CMake verzije 3.16 ili novije
- OpenCV 4.x

Za profilisanje projekta dodatno je potrebno:
- Valgrind (Callgrind)
- KCachegrind (Opciono, za vizuelni prikaz rezultata profilisanja)

## 2. Izgradnja i pokretanje na Linux sistemu.

1. Instalacija potrebnih paketa:
```
sudo apt update
sudo apt install cmake g++ libopencv-dev
```
2. Prelazak u direktorijum u kojem se nalazi izvršna specifikacija:
```
cd spec
```
3. Konfiguracija projekta i njegovo kompajliranje:
```
cmake -S . -B build
cmake --build build --config Release
```
4. Pokretanje programa:
```
./build/HazeRemoval data/train.bmp
```
Programu se kao argument komandne linije prosleđuje putanja do ulazne slike. U ovom primeru se koristi slika train.bmp, koja se nalazi u direktorijumu data. Nakon izvršavanja programa, rekonstruisana slika se prikazuje u prozoru, a takođe se čuva u radnom direktorijumu projekta. 

## 3. Profilisanje izvršne specifikacije

1. Instalacija potrebnih alata:
```
sudo apt install valgrind kcachegrind
```
2.Ponovna konfiguracija projekta i kompajliranje sa informacijama za profilisanje:
```
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-profile -j$(nproc)
```
3. Pokretanje profilisanja:
```
valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./build-profile/HazeRemoval data/train.bmp
```
Nakon završetka izvršavanja programa, rezultati profilisanja biće sačuvani u fajlu callgrind.out.
4. Vizuelni prikaz rezultata:
```
kcachegrind callgrind.out
```