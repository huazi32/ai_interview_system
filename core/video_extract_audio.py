import subprocess
import os

def extract_audio_from_video(
    video_file_path: str,
    output_audio_path: str = "extracted_audio.wav"
) -> str:
    """
    【独立功能】从视频文件中提取声音（音频）
    :param video_file_path: 输入的视频路径
    :param output_audio_path: 输出的音频路径
    :return: 音频文件路径
    """
    try:
        # FFmpeg 命令：提取音频
        command = [
            "ffmpeg",
            "-i", video_file_path,       # 输入视频
            "-vn",                       # 只保留音频
            "-ar", "16000",              # 采样率 16k
            "-ac", "1",                  # 单声道
            "-y",                        # 覆盖已存在文件
            output_audio_path            # 输出音频
        ]

        # 执行提取
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        # 返回生成好的音频路径
        return output_audio_path

    except Exception as e:
        raise Exception(f"视频提取声音失败：{str(e)}")