import os

def find_project_root(current_dir):
    """
    현재 디렉토리에서 시작하여 프로젝트의 루트 디렉토리를 찾습니다.
    특정 파일 또는 디렉토리(예: .git)의 존재를 기준으로 루트를 식별합니다.
    """
    root_marker = '.requirements.txt'  # 루트 디렉토리를 식별할 수 있는 마커
    dir = current_dir
    while True:
        if os.path.exists(os.path.join(dir, root_marker)):
            return dir
        parent_dir = os.path.dirname(dir)
        if parent_dir == dir:
            raise Exception("프로젝트 루트를 찾을 수 없습니다.")
        dir = parent_dir

def to_absolute_path(relative_path):
    """
    주어진 상대 경로를 프로젝트 루트를 기준으로 한 절대 경로로 변환합니다.
    """
    project_root = find_project_root(os.path.dirname(__file__))
    return os.path.join(project_root, relative_path)