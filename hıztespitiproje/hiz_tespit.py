import supervision as sv
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from collections import defaultdict, deque
import time
import matplotlib.pyplot as plt

# Kaynak ve hedef koordinatlarının tanımlanması
SOURCE = np.array([[920, 70], [635, 70], [0, 720], [1440, 720]])
hedefgenislik = 12
hedefyukseklik = 80
hedef = np.array([
    [0, 0],
    [hedefgenislik - 1, 0],
    [hedefgenislik - 1, hedefyukseklik - 1],
    [0, hedefyukseklik - 1],
])

# Görüntü dönüşümü sınıfı
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) != 0:
            reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
            return transformed_points.reshape(-1, 2)
        else:
            return np.array([[1, 2], [5, 8]], dtype='float')

# Ana program
if __name__ == "__main__":
    # Video bilgilerinin alınması
    video_info = sv.VideoInfo.from_video_path(video_path="otoban.mp4")

    # YOLO modelinin yüklenmesi
    model = YOLO("yolov8n.pt")

    # CUDA cihazı seçimi (eğer mevcutsa)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ByteTrack sınıfının oluşturulması
    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=0.5)

    # Etiketleme ve izleme için gerekli parametrelerin tanımlanması
    thickness = 2
    text_scale = 0.55

    # Çerçeve üzerinde kutu çizimi için sınıfın oluşturulması
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)

    # Etiketleme için sınıfın oluşturulması
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )

    # İz sürme için sınıfın oluşturulması
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )

    # Video çerçevelerinin üretildiği sınıfın tanımlanması
    frame_generator = sv.get_video_frames_generator(source_path="otoban.mp4")

    # Bölge tanımlama için çokgen alanın oluşturulması
    polygon_zone = sv.PolygonZone(polygon=SOURCE)

    # Görüntü dönüşümü sınıfının oluşturulması
    view_transformer = ViewTransformer(source=SOURCE, target=hedef)

    # Koordinatlar için varsayılan değerlerin tanımlanması
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Araçların giriş ve çıkış zamanlarının tutulması
    vehicle_times = defaultdict(lambda: {'start': None, 'end': None})

    # Video kaydetmek için gerekli ayarlamalar
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = (648, 1152)
    video = cv2.VideoWriter("cikti_videosu.mp4", cv2_fourcc, 24, size)

    # Son araç algılama zamanının tutulması
    last_detection_time = time.time()

    # Her 20 saniyede bir geçen araç sayısını tutacak değişkenler
    current_interval_start = 0  # Şu anki zaman aralığının başlangıç zamanı
    vehicle_counts = [0]  # Her zaman aralığı için geçen araç sayılarını tutacak liste
    average_times = []  # Her zaman aralığı için ortalama alanda bulunma sürelerini tutacak liste
    average_speeds = []  # Her zaman aralığı için ortalama hızları tutacak liste

    # Ana program döngüsü
    for frame_number, frame in enumerate(frame_generator):
        half_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        result = model(half_frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        # Tespit edilen nesnelerin alt merkez koordinatlarının alınması
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        # Her bir takipçi için koordinatların saklanması
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)
            if vehicle_times[tracker_id]['start'] is None:
                vehicle_times[tracker_id]['start'] = frame_number / video_info.fps
            vehicle_times[tracker_id]['end'] = frame_number / video_info.fps

        # Etiketlerin oluşturulması
        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time_spent = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time_spent * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")

        # Annotasyon yapılmış çerçevenin oluşturulması
        isaretlenmis_kare = half_frame.copy()
        isaretlenmis_kare = sv.draw_polygon(isaretlenmis_kare, polygon=SOURCE, color=sv.Color.RED)
        isaretlenmis_kare = trace_annotator.annotate(scene=isaretlenmis_kare, detections=detections)
        isaretlenmis_kare = bounding_box_annotator.annotate(scene=isaretlenmis_kare, detections=detections)
        isaretlenmis_kare = label_annotator.annotate(scene=isaretlenmis_kare, detections=detections, labels=labels)

        # Araç algılanmazsa "Trafik yok" mesajını göster
        if len(detections) > 0:
            last_detection_time = time.time()
        else:
            if time.time() - last_detection_time > 3:
                cv2.putText(isaretlenmis_kare, "Trafik yok", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Video kaydı
        video.write(isaretlenmis_kare)

        cv2.imshow("frame", isaretlenmis_kare)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Şu anki zamanı al
        current_time = frame_number / video_info.fps

        # Eğer şu anki zaman aralığı sona erdiyse
        if current_time >= current_interval_start + 20:
            # Geçen araç sayısını hesapla
            passed_vehicles = 0
            total_time_spent = 0
            total_speed = 0
            for tracker_id, times in vehicle_times.items():
                if tracker_id in [1, 2, 4]:  # ignore vehicles with IDs 1, 2, and 4
                    continue
                delta_t = times['end'] - times['start']
                if delta_t >= 1:  # 1 saniyeden fazla süren araçları say
                    passed_vehicles += 1
                    total_time_spent += delta_t
                    total_speed += (80 / delta_t) * 3.6

            # Ortalama alanda bulunma süresi ve hız hesaplama
            if passed_vehicles > 0:
                avg_time_spent = total_time_spent / passed_vehicles
                avg_speed = total_speed / passed_vehicles
            else:
                avg_time_spent = 0
                avg_speed = 0

            # Geçen araç sayısını listeye ekle
            vehicle_counts.append(passed_vehicles)
            average_times.append(avg_time_spent)
            average_speeds.append(avg_speed)

            # Yeni zaman aralığını başlat
            current_interval_start = current_time

    # Döngü bittiğinde son zaman aralığının geçen araç sayısını hesapla ve listeye ekle
    passed_vehicles = 0
    total_time_spent = 0
    total_speed = 0
    for tracker_id, times in vehicle_times.items():
        if tracker_id in [1, 2, 4]:  # ignore vehicles with IDs 1, 2, and 4
            continue
        delta_t = times['end'] - times['start']
        if delta_t >= 1:  # 1 saniyeden fazla süren araçları say
            passed_vehicles += 1
            total_time_spent += delta_t
            total_speed += (80 / delta_t) * 3.6
            print(f"Araç {tracker_id}: \t Alanda Bulunma Süresi = {delta_t:.2f} saniye \tOrtalama Hız={(80 / delta_t) * 3.6}")
    vehicle_counts.append(passed_vehicles)
    average_times.append(total_time_spent / passed_vehicles if passed_vehicles > 0 else 0)
    average_speeds.append(total_speed / passed_vehicles if passed_vehicles > 0 else 0)

    # Her 20 saniyedeki araç sayısını hesapla
    vehicle_counts = [vehicle_counts[i + 1] - vehicle_counts[i] for i in range(len(vehicle_counts) - 1)]

    # Sonuçları yazdır
    print("Her 20 saniyede geçen araç sayıları:")
    for i, count in enumerate(vehicle_counts):
        print(f"Zaman Aralığı {i * 20} - {(i + 1) * 20} sn: {count} araç \tOrtalama Alanda Bulunma Süresi = {average_times[i]:.2f} saniye \tOrtalama Hız = {average_speeds[i]:.2f} km/h")

    # Toplamları depolamak için bir liste
    sums = []
    avg_time_sums = []
    avg_speed_sums = []
    densities = []

    # Dizinin uzunluğunu al
    n = len(vehicle_counts)

    # Her ardışık 6 elemanın toplamını hesapla ve 30 ile çarp
    for i in range(n - 5):  # Bu döngü, son 6 elemanı kapsayacak şekilde n-5'e kadar gider
        total_sum = 0
        total_time_sum = 0
        total_speed_sum = 0
        for j in range(6):
            total_sum += vehicle_counts[i + j]
            total_time_sum += average_times[i + j]
            total_speed_sum += average_speeds[i + j]
        total_sum *= (60 / 2)  # Toplamı 30 ile çarp
        avg_time_sum = total_time_sum / 6
        avg_speed_sum = total_speed_sum / 6
        density = total_sum / avg_speed_sum if avg_speed_sum != 0 else 0
        sums.append(total_sum)
        avg_time_sums.append(avg_time_sum)
        avg_speed_sums.append(avg_speed_sum)
        densities.append(density)

    # Sonuçları ekrana yazdır
    for i, sum_value in enumerate(sums):
        print(f"Akım:qa {i + 1} taşıt/saat: {sum_value} \t Ortalama Alanda Bulunma Süresi = {avg_time_sums[i]:.2f} saniye \t Ortalama Hız = {avg_speed_sums[i]:.2f} km/h \t Yoğunluk = {densities[i]:.2f} taşıt/km")

    cv2.destroyAllWindows()
    video.release()

    # Grafikleri çizdirme
    plt.figure(figsize=(12, 8))

    # Akım grafiği
    plt.subplot(3, 1, 1)
    plt.plot(sums, label='Akım (taşıt/saat)', marker='o', linestyle='-')
    plt.title('Akım')
    plt.xlabel('Zaman Aralığı')
    plt.ylabel('Akım (taşıt/saat)')
    plt.grid(True)
    plt.legend()

    # Hız grafiği
    plt.subplot(3, 1, 2)
    plt.plot(avg_speed_sums, label='Ortalama Hız (km/h)', marker='o', linestyle='-')
    plt.title('Ortalama Hız')
    plt.xlabel('Zaman Aralığı')
    plt.ylabel('Ortalama Hız (km/h)')
    plt.grid(True)
    plt.legend()

    # Yoğunluk grafiği
    plt.subplot(3, 1, 3)
    plt.plot(densities, label='Yoğunluk (taşıt/km)', marker='o', linestyle='-')
    plt.title('Yoğunluk')
    plt.xlabel('Zaman Aralığı')
    plt.ylabel('Yoğunluk (taşıt/km)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()