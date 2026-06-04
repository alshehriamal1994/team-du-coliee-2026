# pylint: disable=duplicate-code
import configparser
import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import requests

from src.models.tort import Tort

PATH_TO_ROOT: Final[Path] = Path(__file__).parent.parent
PATH_TO_CONF: Final[Path] = PATH_TO_ROOT / "app.ini"

CONFIG: Final[configparser.ConfigParser] = configparser.ConfigParser()
CONFIG.read(PATH_TO_CONF, encoding="utf-8")

API_KEY: Final[str] = CONFIG["settings"]["API_KEY"]
TEAM: Final[str] = CONFIG["system"]["TEAM_NAME"]
AFFILIATION: Final[str] = CONFIG["system"]["AFFILIATION"]
SYSTEM: Final[str] = CONFIG["system"]["SYSTEM_NAME"]

PATH_TO_DATASET: Final[Path] = PATH_TO_ROOT / "dataset"


BASE_URL: Final[str] = "https://asia-northeast1-ljpjt26.cloudfunctions.net/"
CONNECT_TIMEOUT: Final[float] = 3.0
READ_TIMEOUT: Final[float] = 60.0


def create_url(endpoint: str) -> str:
    return f"{BASE_URL}{endpoint}?key={API_KEY}"


TOKEN_VALIDATOR_URL: Final[str] = create_url("token_validator")
EVALUATION_RESULT_URL: Final[str] = create_url("evaluation_result")
RESULT_UPLOADER_URL: Final[str] = create_url("result_uploader")
DISTRIBUTION_DOWNLOADER_URL: Final[str] = create_url("distribution_downloader")

TEST_DATA_FILENAME: Final[str] = CONFIG["settings"]["TEST_DATA"]
MODE: Final[str] = CONFIG["settings"]["MODE"]
PATH_TO_EVALUATION_RESULTS: Final[Path] = PATH_TO_ROOT / "evaluation_results" / MODE
PATH_TO_SUBMISSIONS: Final[Path] = PATH_TO_ROOT / "submissions" / MODE


def submission_filename() -> str:
    tokens: list[str] = TEST_DATA_FILENAME.split(".")
    return f"{tokens[0]}_{TEAM}_{AFFILIATION}_{SYSTEM}.{tokens[-1]}"


def _log_submission(submission: list[Tort], filename: str) -> None:
    with open(PATH_TO_SUBMISSIONS / filename, mode="w", encoding="utf-8") as f:
        for tort in submission:
            f.write(json.dumps(tort.to_dict(), ensure_ascii=False) + "\n")


def _log_evaluation_results(evaluation_result: dict[str, Any], filename: str) -> None:
    with open(PATH_TO_EVALUATION_RESULTS / filename, mode="w", encoding="utf-8") as f:
        f.write(json.dumps(evaluation_result, ensure_ascii=False) + "\n")


def _download_testdata() -> list[Tort]:
    local_path = PATH_TO_DATASET / TEST_DATA_FILENAME
    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"Using existing local test data: {local_path}")
        return [
            Tort.from_dict(json.loads(line))
            for line in local_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    response = requests.post(
        DISTRIBUTION_DOWNLOADER_URL,
        data=TEST_DATA_FILENAME,
        headers={"Content-Type": "text/plain"},
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    jsonl: str = response.text
    with open(local_path, mode="w", encoding="utf-8") as f:
        f.write(jsonl)
    return [Tort.from_dict(json.loads(line)) for line in jsonl.split("\n")]


def _submit(submission: list[Tort], filename: str, token: str) -> None:
    json_data: dict[str, Any] = {
        "token": token,
        "api_key": API_KEY,
        "filename": filename,
        "mode": MODE,
        "body": [tort.to_dict() for tort in submission],
    }
    text_data: str = json.dumps(json_data)
    requests.post(
        RESULT_UPLOADER_URL,
        data=text_data,
        headers={"Content-Type": "application/json"},
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )


def validate_token(team: str, token: str, is_first: bool) -> tuple[bool, bool]:
    text_data: str = json.dumps({"team": team, "token": token, "is_first": is_first})
    response = requests.post(
        TOKEN_VALIDATOR_URL,
        data=text_data,
        headers={"Content-Type": "application/json"},
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    if response.status_code == 200:
        json_data = response.json()
        is_valid: bool = json_data.get("is_valid", False)
        return is_valid, json_data.get("exceeded_revision_limit", False)
    return False, False


def schedule(base_time: float, interval: int, timeout: int, team: str, token: str) -> bool:
    """
    一定間隔でトークンの有効性をチェックするスケジューラ。
    トークンが無効な間 (False) はループを継続し、有効になったらループを抜ける。

    Args:
        base_time (float): 開始基準時刻 (通常は time.time() を渡す)
        interval (int): 次のスケジュールまでの間隔 (秒)
        timeout (int): タイムアウト (秒)
        team (str): トークン検証で使うチーム情報
        token (str): 検証対象のトークン

    Returns:
        bool: True ならループ継続、False ならループを抜ける
    """
    print(".", end="")

    # NOTE: 経過時間を測定
    elapsed: float = time.time() - base_time
    if timeout < elapsed:
        raise TimeoutError("Timeout")

    # NOTE: 次の実行時刻までの待機時間を計算
    sleep_time: float = interval - (elapsed % interval)
    time.sleep(sleep_time)

    # NOTE: トークンが無効な間ループを回したい
    is_valid, exceeded_revision_limit = validate_token(team, token, is_first=False)
    if exceeded_revision_limit:
        raise ValueError("Exceeded revision limit")
    return not is_valid


def first_check(team: str, token: str) -> bool:
    is_valid, exceeded_revision_limit = validate_token(team, token, is_first=True)
    if exceeded_revision_limit:
        raise ValueError("Exceeded revision limit")
    return is_valid


def evaluate(filename: str, token: str) -> tuple[str, dict[str, Any]]:
    data: str = f"{MODE}/{filename}"
    team: str = filename.split(".")[0].split("_")[1]

    if not first_check(team, token):
        # NOTE: tokenが一致するまで待つ、または、タイムアウトで終了する
        while schedule(  # pylint: disable=while-used
            base_time=time.time(), interval=10, timeout=600, team=team, token=token
        ):
            pass
    print()
    response = requests.post(
        EVALUATION_RESULT_URL,
        data=data,
        headers={"Content-Type": "text/plain"},
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    evaluation_result: dict[str, Any] = {}
    if response.status_code == 200:
        evaluation_result = response.json()

    tokens: list[str] = filename.split(".")
    now: str = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{tokens[0]}_{now}.{tokens[-1]}"

    print(f'The number of revisions: {evaluation_result["num_of_revisions"]}')
    print(f"EVALUATION RESULT: evaluation_results/{MODE}/{filename}")
    print("---")
    print("Tort prediction task")
    print(f"Accuracy: \t{evaluation_result['tort_prediction_task']['accuracy']}")
    print(f"The number of correct answers: \t{evaluation_result['tort_prediction_task']['num_of_correct_answers']}")
    print(f"The number of torts: \t{evaluation_result['tort_prediction_task']['num_of_topics']}")
    print(f"The number of evaluated answers: \t{evaluation_result['tort_prediction_task']['num_of_evaluated_answers']}")
    print("---")
    print("Rationale extraction task")
    print(f"F1 score (all): \t{evaluation_result['rationale_extraction_task']['binary_all_f1']}")
    print(f"Recall (all): \t{evaluation_result['rationale_extraction_task']['binary_all_recall']}")
    print(f"Precision (all): \t{evaluation_result['rationale_extraction_task']['binary_all_precision']}")
    print(f"F1 score (plaintiff): \t{evaluation_result['rationale_extraction_task']['binary_p_f1']}")
    print(f"Recall (plaintiff): \t{evaluation_result['rationale_extraction_task']['binary_p_recall']}")
    print(f"Precision (plaintiff): \t{evaluation_result['rationale_extraction_task']['binary_p_precision']}")
    print(f"F1 score (defendant): \t{evaluation_result['rationale_extraction_task']['binary_d_f1']}")
    print(f"Recall (defendant): \t{evaluation_result['rationale_extraction_task']['binary_d_recall']}")
    print(f"Precision (defendant): \t{evaluation_result['rationale_extraction_task']['binary_d_precision']}")
    print(
        f"The number of correct_answers (all): \t{evaluation_result['rationale_extraction_task']['num_of_all_correct_answers']}"  # noqa: E501
    )
    print(f"The number of claims (all): \t{evaluation_result['rationale_extraction_task']['num_of_all_topics']}")
    print(
        f"The number of evaluated answers (all): \t{evaluation_result['rationale_extraction_task']['num_of_all_evaluated_answers']}"  # noqa: E501
    )
    print(
        f"The number of correct_answers (plaintiff): \t{evaluation_result['rationale_extraction_task']['num_of_p_correct_answers']}"  # noqa: E501
    )
    print(f"The number of claims (plaintiff): \t{evaluation_result['rationale_extraction_task']['num_of_p_topics']}")
    print(
        f"The number of evaluated answers (plaintiff): \t{evaluation_result['rationale_extraction_task']['num_of_p_evaluated_answers']}"  # noqa: E501
    )
    print(
        f"The number of correct_answers (defendant): \t{evaluation_result['rationale_extraction_task']['num_of_d_correct_answers']}"  # noqa: E501
    )
    print(f"The number of claims (defendant): \t{evaluation_result['rationale_extraction_task']['num_of_d_topics']}")
    print(
        f"The number of evaluated answers (defendant): \t{evaluation_result['rationale_extraction_task']['num_of_d_evaluated_answers']}"  # noqa: E501
    )
    return filename, evaluation_result


def pipeline(submission: list[Tort]) -> None:
    filename: str = submission_filename()
    token: str = uuid.uuid1().hex

    _submit(submission, filename, token)

    filename_with_timestamp: str
    evaluation_result: dict[str, Any]
    filename_with_timestamp, evaluation_result = evaluate(filename, token)

    _log_submission(submission, filename_with_timestamp)

    _log_evaluation_results(evaluation_result, filename_with_timestamp)


def main(solve: Callable[[list[Tort]], list[Tort]]) -> None:
    submission: list[Tort] = solve(_download_testdata())
    pipeline(submission)
