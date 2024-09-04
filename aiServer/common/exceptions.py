from rest_framework import status
from rest_framework.exceptions import APIException


class InternalServerError(APIException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = '서버 오류 입니다.'


class BadRequest(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = '잘못된 요청입니다.'
