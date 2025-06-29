import argparse
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Разделение данных на тренировочные, валидационные и тестовые наборы")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Путь к исходному файлу данных (.json, .csv, .txt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Папка для сохранения разделенных данных"
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Доля тренировочных данных (по умолчанию: 0.8)"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Доля валидационных данных (по умолчанию: 0.1)"
    )
        