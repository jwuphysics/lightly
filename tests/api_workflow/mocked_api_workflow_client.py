import csv
import io
import random
import string
import tempfile
import unittest
from io import IOBase
from collections import defaultdict
import json

import numpy as np
from lightly.openapi_generated.swagger_client.api_client import ApiClient

from lightly.openapi_generated.swagger_client.api.quota_api import QuotaApi

from lightly.openapi_generated.swagger_client.api.versioning_api import \
    VersioningApi

from lightly.openapi_generated.swagger_client.api.samples_api import SamplesApi

from lightly.openapi_generated.swagger_client.api.scores_api import ScoresApi
from requests import Response
from lightly.openapi_generated.swagger_client.api.docker_api import DockerApi
from lightly.openapi_generated.swagger_client.model.create_docker_worker_registry_entry_request import CreateDockerWorkerRegistryEntryRequest
from lightly.openapi_generated.swagger_client.model.create_entity_response import \
    CreateEntityResponse
from lightly.openapi_generated.swagger_client.model.dataset_type import \
    DatasetType
from lightly.openapi_generated.swagger_client.model.datasource_processed_until_timestamp_response import DatasourceProcessedUntilTimestampResponse
from lightly.openapi_generated.swagger_client.model.docker_run_data import DockerRunData
from lightly.openapi_generated.swagger_client.model.docker_run_scheduled_create_request import DockerRunScheduledCreateRequest
from lightly.openapi_generated.swagger_client.model.docker_run_scheduled_data import DockerRunScheduledData
from lightly.openapi_generated.swagger_client.model.docker_run_scheduled_priority import DockerRunScheduledPriority
from lightly.openapi_generated.swagger_client.model.docker_run_scheduled_state import DockerRunScheduledState
from lightly.openapi_generated.swagger_client.model.docker_run_state import DockerRunState
from lightly.openapi_generated.swagger_client.model.docker_worker_config_create_request import DockerWorkerConfigCreateRequest
from lightly.openapi_generated.swagger_client.model.docker_worker_registry_entry_data import DockerWorkerRegistryEntryData
from lightly.openapi_generated.swagger_client.model.docker_worker_state import DockerWorkerState
from lightly.openapi_generated.swagger_client.model.docker_worker_type import DockerWorkerType
from lightly.openapi_generated.swagger_client.model.mongo_object_id import \
    MongoObjectID
from lightly.openapi_generated.swagger_client.model.sample_create_request import \
    SampleCreateRequest
from lightly.openapi_generated.swagger_client.model.sample_data import \
    SampleData
from lightly.openapi_generated.swagger_client.model.sample_update_request import \
    SampleUpdateRequest
from lightly.openapi_generated.swagger_client.model.sample_write_urls import \
    SampleWriteUrls
from lightly.openapi_generated.swagger_client.model.tag_arithmetics_request import \
    TagArithmeticsRequest

from lightly.openapi_generated.swagger_client.model.tag_creator import TagCreator
from lightly.openapi_generated.swagger_client.model.dataset_create_request import DatasetCreateRequest
from lightly.openapi_generated.swagger_client.model.dataset_data import DatasetData
from lightly.openapi_generated.swagger_client.api.datasets_api import DatasetsApi
from lightly.openapi_generated.swagger_client.api.datasources_api import DatasourcesApi
from lightly.openapi_generated.swagger_client.model.timestamp import Timestamp
from lightly.openapi_generated.swagger_client.rest import ApiException

import lightly

from lightly.api.api_workflow_client import ApiWorkflowClient

from typing import *

from lightly.openapi_generated.swagger_client.api.embeddings_api import EmbeddingsApi
from lightly.openapi_generated.swagger_client.api.jobs_api import JobsApi
from lightly.openapi_generated.swagger_client.api.mappings_api import MappingsApi
from lightly.openapi_generated.swagger_client.api.samplings_api import SamplingsApi
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi
from lightly.openapi_generated.swagger_client.model.async_task_data import AsyncTaskData
from lightly.openapi_generated.swagger_client.model.dataset_embedding_data import DatasetEmbeddingData
from lightly.openapi_generated.swagger_client.model.job_result_type import JobResultType
from lightly.openapi_generated.swagger_client.model.job_state import JobState
from lightly.openapi_generated.swagger_client.model.job_status_data import JobStatusData
from lightly.openapi_generated.swagger_client.model.job_status_data_result import JobStatusDataResult
from lightly.openapi_generated.swagger_client.model.sampling_create_request import SamplingCreateRequest
from lightly.openapi_generated.swagger_client.model.tag_data import TagData
from lightly.openapi_generated.swagger_client.model.write_csv_url_data import WriteCSVUrlData
from lightly.openapi_generated.swagger_client.model.datasource_config import DatasourceConfig
from lightly.openapi_generated.swagger_client.model.datasource_config_base import DatasourceConfigBase
from lightly.openapi_generated.swagger_client.model.datasource_processed_until_timestamp_request import DatasourceProcessedUntilTimestampRequest
from lightly.openapi_generated.swagger_client.model.datasource_raw_samples_data import DatasourceRawSamplesData
from lightly.openapi_generated.swagger_client.model.datasource_raw_samples_data_row import DatasourceRawSamplesDataRow
from lightly.openapi_generated.swagger_client.model.datasource_raw_samples_predictions_data import DatasourceRawSamplesPredictionsData


def get_random_MongoObjectID() -> str:
    allowed_chars = "abcdef" + string.digits
    id = ''.join(random.choices(allowed_chars, k=24))
    return id


N_FILES_ON_SERVER = 100


class MockedEmbeddingsApi(EmbeddingsApi):
    def __init__(self, api_client):
        EmbeddingsApi.__init__(self, api_client=api_client)
        self.embeddings = [
            DatasetEmbeddingData(
                id=get_random_MongoObjectID(),
                name='embedding_name_xxyyzz',
                isProcessed=True,
                createdAt=0,
            ),
            DatasetEmbeddingData(
                id=get_random_MongoObjectID(),
                name='default',
                isProcessed=True,
                createdAt=0,
            )
        
        ]

    def get_embeddings_csv_write_url_by_id(self, dataset_id: str, **kwargs):
        response_ = WriteCSVUrlData(signed_write_url="signed_write_url_valid", embedding_id=get_random_MongoObjectID())
        return response_

    def get_embeddings_by_dataset_id(self, dataset_id, **kwargs) -> List[DatasetEmbeddingData]:
        return self.embeddings

    def trigger2d_embeddings_job(self, body, dataset_id, embedding_id, **kwargs):
        assert isinstance(body, Trigger2dEmbeddingJobRequest)

    def get_embeddings_csv_read_url_by_id(self, dataset_id, embedding_id, **kwargs):
        return 'https://my-embedding-read-url.com'


class MockedSamplingsApi(SamplingsApi):
    def trigger_sampling_by_id(self, body: SamplingCreateRequest, dataset_id, embedding_id, **kwargs):
        assert isinstance(body, SamplingCreateRequest)
        assert isinstance(dataset_id, str)
        assert isinstance(embedding_id, str)
        response_ = AsyncTaskData(jobId="155")
        return response_


class MockedJobsApi(JobsApi):
    def __init__(self, *args, **kwargs):
        self.no_calls = 0
        JobsApi.__init__(self, *args, **kwargs)

    def get_job_status_by_id(self, job_id, **kwargs):
        assert isinstance(job_id, str)
        self.no_calls += 1
        if self.no_calls > 3:
            result = JobStatusDataResult(type=JobResultType.SAMPLING, data=get_random_MongoObjectID())
            response_ = JobStatusData(id=get_random_MongoObjectID(), status=JobState.FINISHED, waitTimeTillNextPoll=0,
                                      createdAt=1234, finishedAt=1357, result=result)
        else:
            result = None
            response_ = JobStatusData(id=get_random_MongoObjectID(), status=JobState.RUNNING, waitTimeTillNextPoll=0.001,
                                      createdAt=1234, result=result)
        return response_


class MockedTagsApi(TagsApi):
    def create_initial_tag_by_dataset_id(self, body, dataset_id, **kwargs):
        assert isinstance(body, InitialTagCreateRequest)
        response_ = CreateEntityResponse(id="xyz")
        return response_

    def get_tag_by_tag_id(self, dataset_id, tag_id, **kwargs):
        response_ = TagData(id=tag_id, datasetId=dataset_id, prevTagId=get_random_MongoObjectID(), bitMaskData="0x80bda23e9",
                            name='second-tag', totSize=15, createdAt=1577836800, changes=dict())
        return response_

    def get_tags_by_dataset_id(self, dataset_id, **kwargs):
        if dataset_id == 'xyz-no-tags':
            return []
        tag_1 = TagData(id=get_random_MongoObjectID(), datasetId=dataset_id, prevTagId=None,
                        bitMaskData="0xf", name='initial-tag', totSize=4,
                        createdAt=1577836800, changes=dict())
        tag_2 = TagData(id=get_random_MongoObjectID(), datasetId=dataset_id, prevTagId=get_random_MongoObjectID(),
                        bitMaskData="0xf", name='query_tag_name_xyz', totSize=4,
                        createdAt=1577836800, changes=dict())
        tag_3 = TagData(id=get_random_MongoObjectID(), datasetId=dataset_id, prevTagId=get_random_MongoObjectID(),
                        bitMaskData="0x1", name='preselected_tag_name_xyz', totSize=4,
                        createdAt=1577836800, changes=dict())
        tag_4 = TagData(id=get_random_MongoObjectID(), datasetId=dataset_id, prevTagId=get_random_MongoObjectID(),
                        bitMaskData="0x3", name='sampled_tag_xyz', totSize=4,
                        createdAt=1577836800, changes=dict())
        tag_5 = TagData(id=get_random_MongoObjectID(), datasetId=dataset_id, prevTagId=None,
                        bitMaskData='0x1', name='1000', totSize=4,
                        createdAt=1577836800, changes=dict())
        tags = [tag_1, tag_2, tag_3, tag_4, tag_5]
        no_tags_to_return = getattr(self, "no_tags", 5)
        tags = tags[:no_tags_to_return]
        return tags

    def perform_tag_arithmetics(self, body: TagArithmeticsRequest, dataset_id, **kwargs):
        
        if (body.new_tag_name is None) or (body.new_tag_name == ''):
            return TagBitMaskResponse(bitMaskData="0x2")
        else:
            return CreateEntityResponse(id="tag-arithmetic-created")

    def perform_tag_arithmetics_bitmask(self, body: TagArithmeticsRequest, dataset_id, **kwargs):
        
        return TagBitMaskResponse(bitMaskData="0x2")

    def upsize_tags_by_dataset_id(self, body, dataset_id, **kwargs):
        
        assert body.upsize_tag_creator == TagCreator.USER_PIP

    def create_tag_by_dataset_id(self, body, dataset_id, **kwargs):
        
        tag = TagData(id='inital_tag_id', dataset_id=dataset_id, prevTagId=body['prevTagId'],
                      bitMaskData=body['bitMaskData'], name=body['name'], totSize=10,
                      createdAt=1577836800, changes=dict())
        return tag

    def delete_tag_by_tag_id(self, dataset_id, tag_id, **kwargs):
        
        tags = self.get_tags_by_dataset_id(dataset_id)
        # assert that tag exists
        assert any([tag.id == tag_id for tag in tags])
        # assert that tag is a leaf
        assert all([tag.prevTagId != tag_id for tag in tags])

    def export_tag_to_label_studio_tasks(self, dataset_id: str, tag_id: str):
        return [{'id': 0, 'data': {'image': 'https://api.lightly.ai/v1/datasets/62383ab8f9cb290cd83ab5f9/samples/62383cb7e6a0f29e3f31e213/readurlRedirect?type=full&CENSORED', 'lightlyFileName': '2008_006249_jpg.rf.fdd64460945ca901aa3c7e48ffceea83.jpg', 'lightlyMetaInfo': {'type': 'IMAGE', 'datasetId': '62383ab8f9cb290cd83ab5f9', 'fileName': '2008_006249_jpg.rf.fdd64460945ca901aa3c7e48ffceea83.jpg', 'exif': {}, 'index': 0, 'createdAt': 1647852727873, 'lastModifiedAt': 1647852727873, 'metaData': {'sharpness': 27.31265790443818, 'sizeInBytes': 48224, 'snr': 2.1969673926211217, 'mean': [0.24441662557257224, 0.4460417517905863, 0.6960984853824035], 'shape': [167, 500, 3], 'std': [0.12448681278605961, 0.09509570033043004, 0.0763725998175394], 'sumOfSquares': [6282.243860049413, 17367.702452895475, 40947.22059208768], 'sumOfValues': [20408.78823530978, 37244.486274513954, 58124.22352943069]}}}}]


    def export_tag_to_label_box_data_rows(self, dataset_id: str, tag_id: str):
        return [{'externalId': '2008_007291_jpg.rf.2fca436925b52ea33cf897125a34a2fb.jpg', 'imageUrl': 'https://api.lightly.ai/v1/datasets/62383ab8f9cb290cd83ab5f9/samples/62383cb7e6a0f29e3f31e233/readurlRedirect?type=CENSORED'}]


class MockedScoresApi(ScoresApi):
    def create_or_update_active_learning_score_by_tag_id(self, body, dataset_id, tag_id, **kwargs) -> \
            CreateEntityResponse:
        test_scores = np.array(body.scores).astype(float)
        response_ = CreateEntityResponse(id=get_random_MongoObjectID())
        return response_


class MockedMappingsApi(MappingsApi):
    def __init__(self, samples_api, *args, **kwargs):
        self._samples_api = samples_api
        MappingsApi.__init__(self, *args, **kwargs)

        self.n_samples = N_FILES_ON_SERVER
        sample_names = [f'img_{i}.jpg' for i in range(self.n_samples)]
        sample_names.reverse()
        self.sample_names = sample_names
        

    def get_sample_mappings_by_dataset_id(self, dataset_id, field, **kwargs):
        if dataset_id == 'xyz-no-tags':
            return []
        return self.sample_names[:self.n_samples]


class MockedSamplesApi(SamplesApi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_create_requests: List[SampleCreateRequest] = []

    def get_samples_by_dataset_id(
        self, dataset_id, **kwargs
    ) -> List[SampleData]:
        samples = []
        for i, body in enumerate(self.sample_create_requests):
            sample = SampleData(
                id=f'{i}_xyz', 
                dataset_id='dataset_id_xyz', 
                file_name=body.file_name,
                type='Images',
            )
            samples.append(sample)
        return samples

    def create_sample_by_dataset_id(self, body, dataset_id, **kwargs):
        
        assert isinstance(body, SampleCreateRequest)
        response_ = CreateEntityResponse(id="xyz")
        self.sample_create_requests.append(body)
        return response_

    def get_sample_image_write_url_by_id(self, dataset_id, sample_id, is_thumbnail, **kwargs):
        
        url = f"{sample_id}_write_url"
        return url

    def get_sample_image_read_url_by_id(self, dataset_id, sample_id, type, **kwargs):
        
        url = f"{sample_id}_write_url"
        return url

    def get_sample_image_write_urls_by_id(self, dataset_id, sample_id, **kwargs) -> SampleWriteUrls:
        
        thumb_url = f"{sample_id}_thumb_write_url"
        full_url = f"{sample_id}_full_write_url"
        ret = SampleWriteUrls(full=full_url, thumb=thumb_url)
        return ret

    def update_sample_by_id(self, body, dataset_id, sample_id, **kwargs):
        
        assert isinstance(body, SampleUpdateRequest)


class MockedDatasetsApi(DatasetsApi):
    def __init__(self, api_client):
        no_datasets = 3
        self.default_datasets = [
            DatasetData(
                name=f"dataset_{i}", 
                id=get_random_MongoObjectID(),
                lastModifiedAt=i,
                type=DatasetType("Crops"),
                imgType="full",
                sizeInBytes=0,
                nSamples=0,
                createdAt=0,
                userId='user_0',
            )
            for i in range(no_datasets)
        ]
        self.reset()

    def reset(self):
        self.datasets = self.default_datasets

    def get_datasets(self, **kwargs):
        return self.datasets

    def get_all_datasets(self, **kwargs):
        return self.get_datasets()

    def dataset_exists(self, dataset_id: str):
        return dataset_id in [d.id for d in self.default_datasets]

    def create_dataset(self, body: DatasetCreateRequest, **kwargs):
        assert isinstance(body, DatasetCreateRequest)
        id = body.name + "_id"
        if body.name == 'xyz-no-tags':
            id = 'xyz-no-tags'
        dataset = DatasetData(
            id=id, 
            name=body.name, 
            lastModifiedAt=len(self.datasets) + 1,
            type="Images", 
            sizeInBytes=-1, 
            n_samples=-1, 
            createdAt=-1,
            userId='user_0',
        )
        self.datasets += [dataset]
        response_ = CreateEntityResponse(id=id)
        return response_


    def get_dataset_by_id(self, dataset_id):
        
        dataset = next((dataset for dataset in self.default_datasets if dataset_id == dataset.id), None)
        if dataset is None:
            raise ApiException()
        return dataset

    def register_dataset_upload_by_id(self, body, dataset_id):
        
        return True

    def delete_dataset_by_id(self, dataset_id, **kwargs):
        
        datasets_without_that_id = [dataset for dataset in self.datasets if dataset.id != dataset_id]
        assert len(datasets_without_that_id) == len(self.datasets) - 1
        self.datasets = datasets_without_that_id


class MockedDatasourcesApi(DatasourcesApi):
    def __init__(self, api_client=None):
        super().__init__(api_client=api_client)
        # maximum numbers of samples returned by list raw samples request
        self._max_return_samples = 2
        # default number of samples in every datasource
        self._num_samples = 5
        self.reset()

    def reset(self):

        local_datasource = DatasourceConfigBase(type='DatasourceConfigLOCAL', fullPath='', purpose='INPUT')
        azure_datasource = DatasourceConfigBase(type='DatasourceConfigAzure', fullPath='', purpose='INPUT', accountKey="asdf", accountName="asdf")

        self._datasources = {
            "dataset_id_xyz": local_datasource,
            "dataset_0": azure_datasource,
        }
        self._processed_until_timestamp = defaultdict(lambda: 0)
        self._samples = defaultdict(self._default_samples)

    def _default_samples(self):
        return [
            DatasourceRawSamplesDataRow(
                file_name=f"file_{i}", read_url=f"url_{i}"
            )
            for i in range(self._num_samples)
        ]

    def get_datasource_by_dataset_id(self, dataset_id: str, **kwargs):
        try:
            datasource = self._datasources[dataset_id]
            return datasource
        except Exception:
            raise ApiException()

    def get_datasource_processed_until_timestamp_by_dataset_id(
        self, dataset_id: str, **kwargs
    ) -> DatasourceProcessedUntilTimestampResponse:
        timestamp = self._processed_until_timestamp[dataset_id]
        return DatasourceProcessedUntilTimestampResponse(timestamp)

    def get_list_of_raw_samples_from_datasource_by_dataset_id(
        self, dataset_id, cursor: str = None, _from: int = None, to: int = None, **kwargs
    ) -> DatasourceRawSamplesData:
        if cursor is None:
            # initial request
            assert _from is not None
            assert to is not None
            cursor_dict = {"from": _from, "to": to}
            current = _from
        else:
            # follow up request
            cursor_dict = json.loads(cursor)
            current = cursor_dict["current"]
            to = cursor_dict["to"]

        next_current = min(current + self._max_return_samples, to + 1)
        samples = self._samples[dataset_id][current:next_current]
        cursor_dict["current"] = next_current
        cursor = json.dumps(cursor_dict)
        has_more = len(samples) > 0

        return DatasourceRawSamplesData(
            has_more=has_more,
            cursor=cursor,
            data=samples,
        )

    def get_list_of_raw_samples_predictions_from_datasource_by_dataset_id(
        self, dataset_id: str, task_name: str,
        cursor: str = None, _from: int = None, to: int = None, **kwargs,
    ) -> DatasourceRawSamplesPredictionsData:
        if cursor is None:
            # initial request
            assert _from is not None
            assert to is not None
            cursor_dict = {"from": _from, "to": to}
            current = _from
        else:
            # follow up request
            cursor_dict = json.loads(cursor)
            current = cursor_dict["current"]
            to = cursor_dict["to"]

        next_current = min(current + self._max_return_samples, to + 1)
        samples = self._samples[dataset_id][current:next_current]
        cursor_dict["current"] = next_current
        cursor = json.dumps(cursor_dict)
        has_more = len(samples) > 0

        return DatasourceRawSamplesPredictionsData(
            has_more=has_more,
            cursor=cursor,
            data=samples,
        )    


    def get_prediction_file_read_url_from_datasource_by_dataset_id(self, *args, **kwargs):
        return 'https://my-read-url.com'


    def update_datasource_by_dataset_id(
        self, body: DatasourceConfig, dataset_id: str, **kwargs
    ) -> None:
        # TODO: Enable assert once we switch/update to new api code generator.
        # assert isinstance(body, DatasourceConfig)
        self._datasources[dataset_id] = body # type: ignore

    def update_datasource_processed_until_timestamp_by_dataset_id(
        self, body, dataset_id, **kwargs
    ) -> None:
        assert isinstance(body, DatasourceProcessedUntilTimestampRequest)
        to = body.processed_until_timestamp
        self._processed_until_timestamp[dataset_id] = to # type: ignore


class MockedComputeWorkerApi(DockerApi):
    def __init__(self, api_client=None):
        super().__init__(api_client=api_client)
        self._compute_worker_runs = [
            DockerRunData(
                id=get_random_MongoObjectID(),
                dockerVersion="v1",
                dataset_id="dataset_id_xyz",
                state=DockerRunState.TRAINING,
                createdAt=Timestamp(0),
                lastModifiedAt=Timestamp(100),
                message="",
                messages=[],
                report_available=False,
            )
        ]
        self._scheduled_compute_worker_runs = [
            DockerRunScheduledData(
                id=get_random_MongoObjectID(),
                datasetId=get_random_MongoObjectID(),
                configId=get_random_MongoObjectID(),
                priority=DockerRunScheduledPriority.MID,
                state=DockerRunScheduledState.OPEN,
                createdAt=Timestamp(0),
                lastModifiedAt=Timestamp(100),
                owner=get_random_MongoObjectID(),
            )
        ]
        self._registered_workers = [
            DockerWorkerRegistryEntryData(
                id=get_random_MongoObjectID(),
                name="worker-name-1",
                workerType=DockerWorkerType.FULL,
                state=DockerWorkerState.OFFLINE,
                createdAt=Timestamp(0),
                lastModifiedAt=Timestamp(0),
            )
        ]

    def register_docker_worker(self, body, **kwargs):
        assert isinstance(body, CreateDockerWorkerRegistryEntryRequest)
        return CreateEntityResponse(id='worker-id-123')

    def delete_docker_worker_registry_entry_by_id(self, worker_id, **kwargs):
        assert worker_id == 'worker-id-123'

    def get_docker_worker_registry_entries(self, **kwargs):
        return self._registered_workers

    def create_docker_worker_config(self, body, **kwargs):
        assert isinstance(body, DockerWorkerConfigCreateRequest)
        return CreateEntityResponse(id='worker-config-id-123')

    def create_docker_run_scheduled_by_dataset_id(self, body, dataset_id, **kwargs):
        assert isinstance(body, DockerRunScheduledCreateRequest)
        
        return CreateEntityResponse(id=f'scheduled-run-id-123-dataset-{dataset_id}')

    def get_docker_runs(self, **kwargs):
        return self._compute_worker_runs

    def get_docker_runs_scheduled_by_dataset_id(self, dataset_id, **kwargs):
        runs = self._scheduled_compute_worker_runs
        runs = [run for run in runs if run.dataset_id == dataset_id]
        return runs


class MockedVersioningApi(VersioningApi):
    def get_latest_pip_version(self, **kwargs):
        return "1.0.8"

    def get_minimum_compatible_pip_version(self, **kwargs):
        return "1.0.0"


class MockedQuotaApi(QuotaApi):
    def get_quota_maximum_dataset_size(self, **kwargs):
        return "60000"

def mocked_request_put(dst_url: str, data=IOBase) -> Response:
    assert isinstance(dst_url, str)
    content_bytes: bytes = data.read()
    content_str: str = content_bytes.decode('utf-8')
    assert content_str.startswith('filenames')
    response_ = Response()
    response_.status_code = 200
    return response_


class MockedApiClient(ApiClient):
    def request(self, method, url, query_params=None, headers=None,
                post_params=None, body=None, _preload_content=True,
                _request_timeout=None):
        raise ValueError("ERROR: calling ApiClient.request(), but this should be mocked.")

    def call_api(self, resource_path, method,
                 path_params=None, query_params=None, header_params=None,
                 body=None, post_params=None, files=None,
                 response_type=None, auth_settings=None, async_req=None,
                 _return_http_data_only=None, collection_formats=None,
                 _preload_content=True, _request_timeout=None):
        raise ValueError("ERROR: calling ApiClient.call_api(), but this should be mocked.")


class MockedApiWorkflowClient(ApiWorkflowClient):

    embeddings_filename_base = 'img'
    n_embedding_rows_on_server = N_FILES_ON_SERVER

    def __init__(self, *args, **kwargs):
        lightly.api.api_workflow_client.ApiClient = MockedApiClient
        lightly.api.version_checking.VersioningApi = MockedVersioningApi
        ApiWorkflowClient.__init__(self, *args, **kwargs)

        self._selection_api = MockedSamplingsApi(api_client=self.api_client)
        self._jobs_api = MockedJobsApi(api_client=self.api_client)
        self._tags_api = MockedTagsApi(api_client=self.api_client)
        self._embeddings_api = MockedEmbeddingsApi(api_client=self.api_client)
        self._samples_api = MockedSamplesApi(api_client=self.api_client)
        self._mappings_api = MockedMappingsApi(api_client=self.api_client,
                                              samples_api=self._samples_api)
        self._scores_api = MockedScoresApi(api_client=self.api_client)
        self._datasets_api = MockedDatasetsApi(api_client=self.api_client)
        self._datasources_api = MockedDatasourcesApi(api_client=self.api_client)
        self._quota_api = MockedQuotaApi(api_client=self.api_client)
        self._compute_worker_api = MockedComputeWorkerApi(api_client=self.api_client)

        lightly.api.api_workflow_client.requests.put = mocked_request_put

        self.wait_time_till_next_poll = 0.001  # for api_workflow_selection

    def upload_file_with_signed_url(
            self, file: IOBase, signed_write_url: str,
            max_backoff: int = 32, max_retries: int = 5, headers: Dict = None,
    ) -> Response:
        res = Response()
        return res

    def _get_csv_reader_from_read_url(self, read_url: str):
        n_rows: int = self.n_embedding_rows_on_server
        n_dims: int = self.n_dims_embeddings_on_server

        rows_csv = [['filenames'] + [f'embedding_{i}' for i in range(n_dims)] + ['labels']]
        for i in range(n_rows):
            row = [f'{self.embeddings_filename_base}_{i}.jpg']
            for _ in range(n_dims):
                row.append(np.random.uniform(0, 1))
            row.append(i)
            rows_csv.append(row)

        # save the csv rows in a temporary in-memory string file
        # using a csv writer and then read them as bytes
        f = tempfile.SpooledTemporaryFile(mode="rw")
        writer = csv.writer(f)
        writer.writerows(rows_csv)
        f.seek(0)
        buffer = io.StringIO(f.read())
        reader = csv.reader(buffer)

        return reader


class MockedApiWorkflowSetup(unittest.TestCase):
    EMBEDDINGS_FILENAME_BASE: str = 'sample'

    def setUp(self, token="token_xyz",  dataset_id=get_random_MongoObjectID()) -> None:
        self.api_workflow_client = MockedApiWorkflowClient(token=token, dataset_id=dataset_id)
