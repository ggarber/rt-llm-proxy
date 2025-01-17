from abc import abstractmethod
from typing import AsyncIterator, Union
from av import AudioFrame
from PIL.Image import Image

Input = Union[str, AudioFrame, Image]
Output = Union[AudioFrame]


class Model:
    @abstractmethod
    async def send(self, _input: Input):
        pass

    @abstractmethod
    async def recv(self) -> AsyncIterator[Output]:
        pass

    async def close(self):
        pass
