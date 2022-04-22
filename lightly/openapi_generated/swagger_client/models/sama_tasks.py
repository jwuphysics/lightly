# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six

from lightly.openapi_generated.swagger_client.configuration import Configuration


class SamaTasks(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'notification_email': 'str',
        'tasks_file_name': 'str',
        'tasks': 'list[SamaTask]'
    }

    attribute_map = {
        'notification_email': 'notification_email',
        'tasks_file_name': 'tasks_file_name',
        'tasks': 'tasks'
    }

    def __init__(self, notification_email=None, tasks_file_name=None, tasks=None, _configuration=None):  # noqa: E501
        """SamaTasks - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._notification_email = None
        self._tasks_file_name = None
        self._tasks = None
        self.discriminator = None

        if notification_email is not None:
            self.notification_email = notification_email
        if tasks_file_name is not None:
            self.tasks_file_name = tasks_file_name
        self.tasks = tasks

    @property
    def notification_email(self):
        """Gets the notification_email of this SamaTasks.  # noqa: E501


        :return: The notification_email of this SamaTasks.  # noqa: E501
        :rtype: str
        """
        return self._notification_email

    @notification_email.setter
    def notification_email(self, notification_email):
        """Sets the notification_email of this SamaTasks.


        :param notification_email: The notification_email of this SamaTasks.  # noqa: E501
        :type: str
        """

        self._notification_email = notification_email

    @property
    def tasks_file_name(self):
        """Gets the tasks_file_name of this SamaTasks.  # noqa: E501


        :return: The tasks_file_name of this SamaTasks.  # noqa: E501
        :rtype: str
        """
        return self._tasks_file_name

    @tasks_file_name.setter
    def tasks_file_name(self, tasks_file_name):
        """Sets the tasks_file_name of this SamaTasks.


        :param tasks_file_name: The tasks_file_name of this SamaTasks.  # noqa: E501
        :type: str
        """

        self._tasks_file_name = tasks_file_name

    @property
    def tasks(self):
        """Gets the tasks of this SamaTasks.  # noqa: E501


        :return: The tasks of this SamaTasks.  # noqa: E501
        :rtype: list[SamaTask]
        """
        return self._tasks

    @tasks.setter
    def tasks(self, tasks):
        """Sets the tasks of this SamaTasks.


        :param tasks: The tasks of this SamaTasks.  # noqa: E501
        :type: list[SamaTask]
        """
        if self._configuration.client_side_validation and tasks is None:
            raise ValueError("Invalid value for `tasks`, must not be `None`")  # noqa: E501

        self._tasks = tasks

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(SamaTasks, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, SamaTasks):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, SamaTasks):
            return True

        return self.to_dict() != other.to_dict()
