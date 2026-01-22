import os
import shutil

def delete_pycache_folders(start_path):
    """
    Рекурсивно находит и удаляет все папки __pycache__
    в указанной директории.
    """
    if not os.path.isdir(start_path):
        print(f"Ошибка: Указанный путь '{start_path}' не является директорией.")
        return

    print(f"Поиск и удаление папок __pycache__ в: {start_path}")
    deleted_count = 0
    for root, dirs, files in os.walk(start_path):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"Удалено: {pycache_path}")
                deleted_count += 1
            except OSError as e:
                print(f"Ошибка при удалении {pycache_path}: {e}")

    if deleted_count > 0:
        print(f"\nУспешно удалено {deleted_count} папок __pycache__.")
    else:
        print("\nПапки __pycache__ не найдены.")

if __name__ == "__main__":
    # По умолчанию скрипт очищает директорию, в которой он запущен.
    # Чтобы очистить конкретную папку, замените os.getcwd() на ваш путь, например:
    # folder_to_clean = "/path/to/your/project"
    folder_to_clean = os.getcwd()
    delete_pycache_folders(folder_to_clean)