from dataclasses import dataclass, asdict, field
import datetime


@dataclass
class ResponseWrapper:
    status: int = field(init=False)
    message: str = field(init=False)
    data: object = None
    time: datetime.datetime = field(default_factory=datetime.datetime.now, init=False)

    def __post_init__(self):
        self.time = datetime.datetime.now()

    def to_dict(self):
        # 将dataclasses实例转换为字典
        return asdict(self)

    @classmethod
    def success(cls):
        return cls._create_response(1, "Success.")

    @classmethod
    def failed(cls):
        return cls._create_response(0, "Failed!")

    @classmethod
    def error(cls):
        return cls._create_response(-1, "Error!")

    @staticmethod
    def _create_response(status, message, data=None):
        wrapper = ResponseWrapper()
        wrapper.status = status
        wrapper.message = message
        wrapper.data = data
        return wrapper

    def set_message(self, message):
        self.message = message
        return self

    def set_data(self, data):
        self.data = data
        return self

    def is_success(self):
        return self.status == 1

    def is_failed(self):
        return self.status == 0

    def is_error(self):
        return self.status == -1
