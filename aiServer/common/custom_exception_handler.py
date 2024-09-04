import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler
from django.core.exceptions import ValidationError

logger = logging.getLogger('django')


def custom_exception_handler(exc, context):
    logger.error("custom_exception_handler() : {0}".format(str(exc)))

    if isinstance(exc, ValueError):
        response_data = {'isSuccess': False, 'detail': 'Bad request: Invalid value provided.', 'status_code': status.HTTP_400_BAD_REQUEST}
        return Response(response_data, status=status.HTTP_400_BAD_REQUEST)

    if isinstance(exc, ValidationError):
        response_data = {'isSuccess': False, 'detail': 'Bad request: Invalid value provided.', 'status_code': status.HTTP_400_BAD_REQUEST}
        return Response(response_data, status=status.HTTP_400_BAD_REQUEST)

    response = exception_handler(exc, context)

    if response is not None:
        response.data['status_code'] = response.status_code

    return response
