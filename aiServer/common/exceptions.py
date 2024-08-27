from rest_framework import status
from rest_framework.exceptions import APIException


class ApiKeyForbidden(APIException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = '서버 오류 입니다.'
