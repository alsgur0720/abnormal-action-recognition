from PIL import Image

def extract_and_save_frames(gif_path, output_folder):
    with Image.open(gif_path) as img:
        # frame_count를 확인하여 GIF의 프레임 수를 확인
        frame_count = img.n_frames
        print(f"Total frames: {frame_count}")
        
        # 각 프레임을 돌면서 이미지 파일로 저장
        for i in range(frame_count):
            img.seek(i)  # i번째 프레임으로 이동
            img.save(f"{output_folder}/frame_{i}.png")  # 프레임을 PNG로 저장

# 사용 예
gif_path = './motion_test/motion_test_hand_waving.gif'
output_folder = './projcet_eg'
extract_and_save_frames(gif_path, output_folder)
