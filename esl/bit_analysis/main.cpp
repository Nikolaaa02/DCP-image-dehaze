#include "hazeremoval.h"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <cstdint>
#include <limits>
#include <sstream>

using namespace cv;
using namespace std;

struct FixedPointTestStats {
    double absErrorSum = 0.0;
    double squaredErrorSum = 0.0;
    double maxAbsError = 0.0;

    uint64_t samples = 0;
    uint64_t overflowCount = 0;
};

// Agregirani kvalitet konacne rekonstruisane slike preko vise ulaznih slika.
// MAE i PSNR se na kraju racunaju iz ukupnih gresaka svih RGB kanala,
// tako da vece slike proporcionalno doprinose vecem broju uzoraka.
struct ImageQualityAggregateStats {
    // Ukupne greske svih slika, za zbirni MAE i PSNR.
    double absoluteErrorSum = 0.0;
    double squaredErrorSum = 0.0;
    uint64_t samples = 0;
    uint64_t images = 0;

    // Najgori rezultat medju pojedinacnim slikama.
    double maxImageMAE = 0.0;
    double minImagePSNR = std::numeric_limits<double>::infinity();

    // Najveca razlika RGB kanala preko celog skupa.
    int maxDifference = 0;

    // Nazivi slika koje imaju najgori MAE i PSNR.
    string maxMAEImage;
    string minPSNRImage;
};

static uint64_t roundDivideUnsigned(uint64_t numerator,
    uint64_t denominator)
{
    return (numerator + denominator / 2) / denominator;
}

static bool simulateTransmissionFixed(
    const cv::Mat& src,
    const cv::Vec3d& atmosphericLight,
    const cv::Mat& referenceTransmission,
    cv::Mat& fixedTransmission,
    int radius,
    double omega,
    int F,
    FixedPointTestStats& stats)
{
    // scale predstavlja 2^F.
    const uint64_t scale = uint64_t{ 1 } << F;

    // Neoznaceni Q8.F za atmosfersko svetlo.
    const uint64_t maxA = (uint64_t{ 1 } << 8) * scale - 1;
    uint64_t Araw[3];

    for (int c = 0; c < 3; ++c) {
        const double a = atmosphericLight[c];

        if (!std::isfinite(a) || a < 0.0 || a > 255.0) {
            ++stats.overflowCount;
            return false;
        }

        Araw[c] = static_cast<uint64_t>(
            std::llround(a * static_cast<double>(scale)));

        if (Araw[c] > maxA) {
            ++stats.overflowCount;
            Araw[c] = maxA;
        }

        if (Araw[c] == 0)
            return false;
    }

    // Neoznaceni Q1.F za omega.
    const uint64_t omegaRaw = static_cast<uint64_t>(
        std::llround(omega * static_cast<double>(scale)));

    // Neoznaceni Q7.F za odnose B/A, G/A i R/A.
    const uint64_t maxUnsignedQ7 =
        (uint64_t{ 1 } << 7) * scale - 1;

    // Neoznaceni Q4.F za minpixel.
    const uint64_t maxUnsignedQ4 =
        (uint64_t{ 1 } << 4) * scale - 1;

    // Neoznaceni Q3.F za min_val.
    const uint64_t maxUnsignedQ3 =
        (uint64_t{ 1 } << 3) * scale - 1;

    // Oznaceni Q3.F za transmission: opseg [-4, 4 - 2^-F].
    const int64_t minSignedQ3 =
        -static_cast<int64_t>(uint64_t{ 1 } << 2) *
        static_cast<int64_t>(scale);

    const int64_t maxSignedQ3 =
        static_cast<int64_t>(
            (uint64_t{ 1 } << 2) * scale - 1);

    fixedTransmission.create(
        src.rows, src.cols, CV_64FC1);

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {

            const int rmin = std::max(0, i - radius);
            const int rmax = std::min(i + radius, src.rows - 1);
            const int cmin = std::max(0, j - radius);
            const int cmax = std::min(j + radius, src.cols - 1);

            // Odnosi B/A, G/A i R/A u Q7.F.
            // MinPixel i minVal cuvaju se kao celobrojni kodovi.
            uint64_t minValRaw = maxUnsignedQ3;

            for (int x = rmin; x <= rmax; ++x) {
                for (int y = cmin; y <= cmax; ++y) {
                    const cv::Vec3b pixel =
                        src.ptr<cv::Vec3b>(x)[y];

                    uint64_t ratioRaw[3];

                    for (int c = 0; c < 3; ++c) {
                        // pixel[c] / A[c], zaokruzeno u Q2.F.
                        // Araw[c] = A[c] * scale.
                        const uint64_t numerator =
                            static_cast<uint64_t>(pixel[c]) *
                            scale * scale;

                        uint64_t q = roundDivideUnsigned(
                            numerator, Araw[c]);

                        if (q > maxUnsignedQ7) {
                            ++stats.overflowCount;
                            q = maxUnsignedQ7;
                        }

                        ratioRaw[c] = q;
                    }

                    uint64_t minPixelRaw =
                        std::min(ratioRaw[0],
                            std::min(ratioRaw[1], ratioRaw[2]));

                    // UQ4.F ogranicenje za minpixel.
                    if (minPixelRaw > maxUnsignedQ4) {
                        ++stats.overflowCount;
                        minPixelRaw = maxUnsignedQ4;
                    }

                    // UQ3.F ogranicenje za min_val.
                    minValRaw = std::min(minValRaw, minPixelRaw);
                }
            }

            // Proizvod omega * minVal u Q3.F.
            const uint64_t productRaw =
                roundDivideUnsigned(
                    omegaRaw * minValRaw, scale);

            uint64_t boundedProduct = productRaw;

            if (boundedProduct > maxUnsignedQ3) {
                ++stats.overflowCount;
                boundedProduct = maxUnsignedQ3;
            }

            // t = 1 - product, oznaceni Q3.F.
            int64_t transmissionRaw =
                static_cast<int64_t>(scale) -
                static_cast<int64_t>(boundedProduct);

            if (transmissionRaw < minSignedQ3) {
                ++stats.overflowCount;
                transmissionRaw = minSignedQ3;
            }

            if (transmissionRaw > maxSignedQ3) {
                ++stats.overflowCount;
                transmissionRaw = maxSignedQ3;
            }

            // Pretvaranje u double sluzi samo za merenje greske.
            const double transmission =
                static_cast<double>(transmissionRaw) /
                static_cast<double>(scale);

            fixedTransmission.at<double>(i, j) =
                transmission;

            const double reference =
                referenceTransmission.at<double>(i, j);

            const double error =
                std::abs(transmission - reference);

            stats.absErrorSum += error;
            stats.squaredErrorSum += error * error;
            stats.maxAbsError =
                std::max(stats.maxAbsError, error);

            ++stats.samples;
        }
    }

    return true;
}

static cv::Mat makeDehazedImage(
    const cv::Mat& image,
    const cv::Mat& transmission,
    cv::Vec3d& atmosphericLight,
    int guidedRadius,
    double guidedEps,
    double t0)
{
    cv::Mat guidedTransmission;
    cv::Mat recovered(
        image.rows, image.cols, CV_64FC3);

    // 1. Guided filter: preciscavanje transmisije.
    guided_filter(
        &image,
        &transmission,
        &guidedTransmission,
        guidedRadius,
        guidedEps);

    // 2. Rekonstrukcija slike.
    recover(
        &image,
        &guidedTransmission,
        &recovered,
        &atmosphericLight,
        image.rows,
        image.cols,
        image.channels(),
        t0);

    // 3. Konverzija u 8-bitnu sliku
    cv::Mat result(
        image.rows, image.cols, CV_8UC3);

    for (int i = 0; i < image.rows; ++i) {
        const cv::Vec3d* src =
            recovered.ptr<cv::Vec3d>(i);

        cv::Vec3b* dst =
            result.ptr<cv::Vec3b>(i);

        for (int j = 0; j < image.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                dst[j][c] =
                    static_cast<uchar>(src[j][c]);
            }
        }
    }

    return result;
}


static void compareImages(
    const cv::Mat& reference,
    const cv::Mat& test,
    double& mae,
    double& psnr,
    int& maxDifference,
    ImageQualityAggregateStats* aggregate = nullptr)
{
    CV_Assert(reference.size() == test.size());
    CV_Assert(reference.type() == CV_8UC3);
    CV_Assert(test.type() == CV_8UC3);

    double absoluteErrorSum = 0.0;
    double squaredErrorSum = 0.0;

    uint64_t samples = 0;
    maxDifference = 0;

    for (int i = 0; i < reference.rows; ++i) {
        const cv::Vec3b* ref =
            reference.ptr<cv::Vec3b>(i);

        const cv::Vec3b* tst =
            test.ptr<cv::Vec3b>(i);

        for (int j = 0; j < reference.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                const int difference =
                    static_cast<int>(ref[j][c]) -
                    static_cast<int>(tst[j][c]);

                const int absDifference =
                    std::abs(difference);

                absoluteErrorSum += absDifference;
                squaredErrorSum +=
                    static_cast<double>(difference) *
                    difference;

                maxDifference = std::max(
                    maxDifference, absDifference);

                ++samples;
            }
        }
    }

    mae = absoluteErrorSum / samples;

    const double mse = squaredErrorSum / samples;

    if (mse == 0.0) {
        psnr = std::numeric_limits<double>::infinity();
    }
    else {
        psnr = 10.0 * std::log10(
            (255.0 * 255.0) / mse);
    }

    // Ako je prosledjen agregator, saberi greske ove slike u ukupnu statistiku.
    if (aggregate != nullptr) {
        aggregate->absoluteErrorSum += absoluteErrorSum;
        aggregate->squaredErrorSum += squaredErrorSum;
        aggregate->samples += samples;
        aggregate->maxDifference = std::max(
            aggregate->maxDifference, maxDifference);
        ++aggregate->images;
    }
}


static void printRange(
    const string& name,
    const BitRangeStats& range)
{
    cout << name
        << " | min: " << range.minValue
        << " [" << range.minImage << "]"
        << " | max: " << range.maxValue
        << " [" << range.maxImage << "]"
        << '\n';
}

static void writeRange(
    ofstream& file,
    const string& name,
    const BitRangeStats& range)
{
    file << name << ','
        << range.minValue << ",\""
        << range.minImage << "\","
        << range.maxValue << ",\""
        << range.maxImage << "\"\n";
}

static void addImages(
    const string& folder,
    const vector<string>& extensions,
    vector<String>& files)
{
    for (const string& ext : extensions) {
        vector<String> matches;

        // true znaci da se pretrazuju i podfolderi.
        glob(folder + "*." + ext, matches, true);

        files.insert(
            files.end(),
            matches.begin(),
            matches.end());
    }
}

static string normalizePath(string path)
{
    replace(path.begin(), path.end(), '\\', '/');
    return path;
}

static bool isSelectedImage(
    const string& imagePath,
    const string& selectedImagePath)
{
    const string image = normalizePath(imagePath);
    const string selected = normalizePath(selectedImagePath);

    // Potpuno poklapanje putanja.
    if (image == selected)
        return true;

    // Omogucava da se prosledi samo naziv slike
    // ili relativni deo putanje.
    if (image.size() <= selected.size())
        return false;

    const size_t start = image.size() - selected.size();

    return image.compare(start, selected.size(), selected) == 0
        && image[start - 1] == '/';
}

static bool parseFracBits(
    const string& input,
    vector<int>& fracBits)
{
    vector<int> parsed;
    stringstream ss(input);
    string token;

    try {
        while (getline(ss, token, ',')) {

            // Uklanjanje eventualnih razmaka.
            const size_t first = token.find_first_not_of(" \t");
            const size_t last = token.find_last_not_of(" \t");

            if (first == string::npos)
                return false;

            token = token.substr(first, last - first + 1);

            size_t pos = 0;
            int F = stoi(token, &pos);

            if (pos != token.size() || F < 0 || F > 28)
                return false;

            // Ne dodajemo iste vrednosti vise puta.
            if (find(parsed.begin(), parsed.end(), F) == parsed.end())
                parsed.push_back(F);
        }
    }
    catch (...) {
        return false;
    }

    if (parsed.empty())
        return false;

    fracBits = parsed;
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        cerr << "Upotreba: HazeRemoval_BitAnalysis.exe "
            << "\"putanja_do_foldera\" "
            << "[--image \"putanja_do_slike\"] "
            << "[--frac-bits 7,8,9,10,11]\n";

        return 1;
    }

    // Putanja do celog skupa slika ostaje prvi argument.
    string folder = argv[1];

    // Podrazumevana slika za cuvanje vizuelnih rezultata.
    // Moze biti samo naziv slike ili relativna putanja unutar skupa.
    string visualizeImage = "SGP_Google_345.png";

    // Podrazumevani kandidati za broj razlomljenih bitova.
    vector<int> fracBitsCandidates = { 7, 8, 9, 10, 11 };

    // Obrada opcionalnih argumenata.
    for (int arg = 2; arg < argc; ++arg) {

        string option = argv[arg];

        if (option == "--image") {

            if (arg + 1 >= argc) {
                cerr << "Nedostaje putanja do slike nakon --image.\n";
                return 1;
            }

            visualizeImage = argv[++arg];
        }

        else if (option == "--frac-bits") {

            if (arg + 1 >= argc) {
                cerr << "Nedostaju vrednosti F nakon --frac-bits.\n";
                return 1;
            }

            string values = argv[++arg];

            if (!parseFracBits(values, fracBitsCandidates)) {
                cerr << "Neispravne vrednosti F: " << values << '\n';
                cerr << "Dozvoljene su celobrojne vrednosti od 0 do 28, "
                    << "razdvojene zarezima.\n";
                return 1;
            }
        }

        else {
            cerr << "Nepoznata opcija: " << option << '\n';
            return 1;
        }
    }

    // Ujednaci separatore putanje na Windows-u.
    replace(folder.begin(), folder.end(), '\\', '/');

    if (!folder.empty() && folder.back() != '/') {
        folder += '/';
    }

    // Normalizacija putanje iz argumenta.
    visualizeImage = normalizePath(visualizeImage);

    vector<string> extensions = {
        "jpg", "jpeg", "png", "bmp", "tif", "tiff",
        "JPG", "JPEG", "PNG", "BMP", "TIF", "TIFF"
    };

    vector<String> files;
    addImages(folder, extensions, files);

    sort(files.begin(), files.end());
    files.erase(
        unique(files.begin(), files.end()),
        files.end());

    if (files.empty()) {
        cerr << "Nisu pronadjene slike u folderu: "
            << folder << '\n';
        return 1;
    }

    cout << "Pronadjeno slika: " << files.size() << '\n';

    DatasetBitStats globalStats;

    const int radius = 7;
    const double omega = 0.95;

    size_t processed = 0;
    size_t skipped = 0;

    std::map<int, FixedPointTestStats> fixedStats;
    std::map<int, ImageQualityAggregateStats> finalImageStats;

    // CSV sa kvalitetom konacne slike za svaku kombinaciju (ulazna slika, F).
    ofstream imageQualityCsv("fixed_point_image_quality.csv");
    if (imageQualityCsv.is_open()) {
        imageQualityCsv << "image,F,MAE,PSNR_dB,max_pixel_difference\n";
        imageQualityCsv << setprecision(15);
    }
    else {
        cerr << "Upozorenje: ne mogu da otvorim "
            << "fixed_point_image_quality.csv za upis.\n";
    }

    // Parametri istog guided filtera i rekonstrukcije kao u originalnom Process() postupku.
    const int guidedRadius = 60;
    const double guidedEps = 0.001;
    const double t0 = 0.1;

    for (size_t i = 0; i < files.size(); ++i) {
        const string filename = files[i];

        Mat image = imread(filename, IMREAD_COLOR);

        if (image.empty()) {
            cerr << "Preskacem neispravnu sliku: "
                << filename << '\n';
            ++skipped;
            continue;
        }

        if (image.total() < 1000) {
            cerr << "Preskacem premalu sliku: "
                << filename << '\n';
            ++skipped;
            continue;
        }

        const int rows = image.rows;
        const int cols = image.cols;
        const int channels = image.channels();

        // Potrebno za lokalnu mapu transmisije.
        Mat transmission(rows, cols, CV_64FC1);

        // Za svaku sliku pravimo novi vektor kandidata.
        vector<Pixel> candidates;

        Vec3d atmosphericLight(0.0, 0.0, 0.0);

        // 1. Dark channel je potreban za procenu A.
        get_dark_channel(
            &image,
            candidates,
            rows,
            cols,
            channels,
            radius);

        // 2. Procena atmosferskog svetla.
        get_air_light(
            &image,
            candidates,
            &atmosphericLight,
            rows,
            cols,
            channels);

        // Provera validnosti atmosferskog svetla.
        bool validAtmosphericLight = true;

        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(atmosphericLight[c]) ||
                atmosphericLight[c] <= 0.0) {

                validAtmosphericLight = false;
                break;
            }
        }

        if (!validAtmosphericLight) {
            cerr << "Preskacem sliku zbog nevalidnog atmosferskog svetla: "
                << filename
                << " | A (B, G, R) = "
                << atmosphericLight[0] << ", "
                << atmosphericLight[1] << ", "
                << atmosphericLight[2] << '\n';

            ++skipped;
            continue;
        }

        globalStats.airlightB.update(
            atmosphericLight[0],
            atmosphericLight[0],
            filename);

        globalStats.airlightG.update(
            atmosphericLight[1],
            atmosphericLight[1],
            filename);

        globalStats.airlightR.update(
            atmosphericLight[2],
            atmosphericLight[2],
            filename);

        // 3. Racunanje transmisije i azuriranje globalnih opsega za ovu sliku.
        get_transmission_measurement(
            &image,
            &transmission,
            &atmosphericLight,
            rows,
            cols,
            channels,
            radius,
            omega,
            &globalStats,
            filename,
            false);

        const bool createVisualsThisImage =
            isSelectedImage(filename, visualizeImage);

        if (createVisualsThisImage) {
            std::cout << "\nPronasao izabranu sliku: "
                << filename << '\n';
        }

        // Referentnu (double) konacnu sliku racunamo za svaku ulaznu sliku.
        // Sacuvacemo je na disk samo za izabrani primer, da ne pravimo gomilu izlaznih PNG fajlova.
        const cv::Mat referenceDehazed = makeDehazedImage(
            image,
            transmission,
            atmosphericLight,
            guidedRadius,
            guidedEps,
            t0);

        if (createVisualsThisImage) {
            cv::imwrite("reference_double.png", referenceDehazed);
            std::cout << "\nReferentna slika sacuvana: reference_double.png\n";
        }

        // Sada se svaki kandidat F testira na svakoj uspesno ucitanoj slici.
        for (int F : fracBitsCandidates) {
            cv::Mat fixedTransmission;

            const bool ok = simulateTransmissionFixed(
                image,
                atmosphericLight,
                transmission,
                fixedTransmission,
                radius,
                omega,
                F,
                fixedStats[F]);

            if (!ok) {
                std::cerr
                    << "Neispravno atmosfersko svetlo za F = "
                    << F << ", slika: " << filename << '\n';
                continue;
            }

            // Guided filter i recover se izvrsavaju nad transmisijom dobijenom za ovaj F, pa poredimo konacne RGB slike.
            const cv::Mat fixedDehazed = makeDehazedImage(
                image,
                fixedTransmission,
                atmosphericLight,
                guidedRadius,
                guidedEps,
                t0);

            double imageMAE;
            double imagePSNR;
            int maxDifference;

            compareImages(
                referenceDehazed,
                fixedDehazed,
                imageMAE,
                imagePSNR,
                maxDifference,
                &finalImageStats[F]);

            // Azuriranje najgorih rezultata medju pojedinacnim slikama.
            ImageQualityAggregateStats& qualityStats = finalImageStats[F];

            // Najveci MAE pojedinacne slike.
            if (qualityStats.maxMAEImage.empty() ||
                imageMAE > qualityStats.maxImageMAE) {

                qualityStats.maxImageMAE = imageMAE;
                qualityStats.maxMAEImage = filename;
            }

            // Najmanji PSNR pojedinacne slike.
            if (qualityStats.minPSNRImage.empty() ||
                imagePSNR < qualityStats.minImagePSNR) {

                qualityStats.minImagePSNR = imagePSNR;
                qualityStats.minPSNRImage = filename;
            }

            // maxDifference se vec azurira unutar funkcije compareImages().

            // Sacuvaj metrikе za svaku pojedinacnu sliku i svaki F.
            if (imageQualityCsv.is_open()) {
                imageQualityCsv << '"' << filename << '"' << ','
                    << F << ','
                    << imageMAE << ','
                    << imagePSNR << ','
                    << maxDifference << '\n';
            }

            // Fajlove slika i detaljan ispis cuvamo samo za odabranu sliku.
            if (createVisualsThisImage) {
                const std::string outputName =
                    "fixed_F" + std::to_string(F) + ".png";
                cv::imwrite(outputName, fixedDehazed);

                std::cout << "\nPoredjenje konacne slike za F = "
                    << F << '\n';
                std::cout << "MAE slike: " << imageMAE << '\n';
                std::cout << "PSNR slike: " << imagePSNR << " dB\n";
                std::cout << "Maksimalna razlika piksela: "
                    << maxDifference << '\n';
                std::cout << "Sacuvana slika: " << outputName << '\n';
            }
        }

        ++processed;

        cout << "[" << processed << "/" << files.size()
            << "] Obradjeno: " << filename << '\n';
    }

    if (processed == 0) {
        cerr << "Nijedna slika nije uspesno obradjena.\n";
        return 1;
    }

    cout << "\n===== GLOBALNA BITSKA ANALIZA =====\n";
    cout << "Uspesno obradjeno: " << processed << '\n';
    cout << "Preskoceno: " << skipped << '\n';
    cout << setprecision(12);

    // Atmosfersko svetlo
    printRange("A_B", globalStats.airlightB);
    printRange("A_G", globalStats.airlightG);
    printRange("A_R", globalStats.airlightR);

    // Normalizovani kanali
    printRange("B/A", globalStats.bRatio);
    printRange("G/A", globalStats.gRatio);
    printRange("R/A", globalStats.rRatio);

    // Ostali medjurezultati
    printRange("minpixel", globalStats.minpixel);
    printRange("min_val", globalStats.minVal);
    printRange("transmission", globalStats.transmission);

    ofstream csv("bit_analysis_global.csv");

    if (csv.is_open()) {
        csv << setprecision(15);
        csv << "metric,min_value,min_image,max_value,max_image\n";

        // Atmosfersko svetlo
        writeRange(csv, "A_B", globalStats.airlightB);
        writeRange(csv, "A_G", globalStats.airlightG);
        writeRange(csv, "A_R", globalStats.airlightR);

        // Normalizovani kanali i ostali medjurezultati
        writeRange(csv, "B/A", globalStats.bRatio);
        writeRange(csv, "G/A", globalStats.gRatio);
        writeRange(csv, "R/A", globalStats.rRatio);
        writeRange(csv, "minpixel", globalStats.minpixel);
        writeRange(csv, "min_val", globalStats.minVal);
        writeRange(csv, "transmission", globalStats.transmission);

        csv.close();

        cout << "\nRezultati sacuvani u "
            << "bit_analysis_global.csv\n";
    }
    else {
        cerr << "\nNe mogu da sacuvam CSV fajl.\n";
    }

    if (imageQualityCsv.is_open()) {
        imageQualityCsv.close();
        cout << "\nDetaljni rezultati konacnih slika sacuvani u "
            << "fixed_point_image_quality.csv\n";
    }

    std::cout
        << "\n===== SIMULACIJA FIKSNOG ZAREZA =====\n";

    for (int F : fracBitsCandidates) {
        const FixedPointTestStats& s = fixedStats[F];

        if (s.samples == 0)
            continue;

        const double mae =
            s.absErrorSum / s.samples;

        const double rmse =
            std::sqrt(s.squaredErrorSum / s.samples);

        std::cout << "\nF = " << F << '\n';
        std::cout << "MAE transmisije: " << mae << '\n';
        std::cout << "RMSE transmisije: " << rmse << '\n';

        std::cout << "Maksimalna apsolutna greska: "
            << s.maxAbsError << '\n';

        std::cout << "Broj prekoracenja u medjurezultatima: "
            << s.overflowCount << '\n';
    }

    std::cout << "\n===== KVALITET KONACNE SLIKE PREKO CELOG SKUPA =====\n";
    for (int F : fracBitsCandidates) {
        const ImageQualityAggregateStats& s = finalImageStats[F];
        if (s.samples == 0)
            continue;

        const double datasetMAE = s.absoluteErrorSum / s.samples;
        const double datasetMSE = s.squaredErrorSum / s.samples;
        const double datasetPSNR = (datasetMSE == 0.0)
            ? std::numeric_limits<double>::infinity()
            : 10.0 * std::log10((255.0 * 255.0) / datasetMSE);

        std::cout << "\nF = " << F << '\n';
        std::cout << "Uspesno uporedjenih slika: " << s.images << '\n';
        std::cout << "MAE (sve slike, svi RGB kanali): " << datasetMAE << '\n';
        std::cout << "PSNR (iz ukupnog MSE): " << datasetPSNR << " dB\n";
        std::cout << "Najveci MAE pojedinacne slike: " << s.maxImageMAE << '\n';
        std::cout << "Slika sa najvecim MAE: " << s.maxMAEImage << '\n';
        std::cout << "Najmanji PSNR pojedinacne slike: " << s.minImagePSNR << " dB\n";
        std::cout << "Slika sa najmanjim PSNR: " << s.minPSNRImage << '\n';
        std::cout << "Maksimalna razlika piksela preko skupa: " << s.maxDifference << '\n';
    }

    return 0;
}