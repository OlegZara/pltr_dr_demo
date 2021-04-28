from transforms.api import transform_df, Input, Output
import datarobot as dr


@transform_df(
    Output("ri.foundry.main.dataset.f424168e-8fc8-4048-a35c-c8fb9f8be430"),
    modelling_data=Input('ri.foundry.main.dataset.3af76283-ed7e-403c-9bd4-df7e6d0b14a2'),
    credentials=Input('ri.foundry.main.dataset.0c96802f-03dd-4ab6-a824-5fd1b8d0d32f')
)
def train_model(modelling_data, credentials):
    '''
    This function submits a dataset to DataRobot for model build
    :param modelling_data: DataFrame containing training data
    :param credentials: DataFrame with users token
    :return: deployment meta data
    '''
    # Configure the client by loading private DataRobot token:
    credentials = credentials.toPandas()
    credentials = dict(zip(credentials.key, credentials.value))

    TOKEN = credentials['TOKEN']
    ENDPOINT = 'https://app.datarobot.com/api/v2'
    dr.Client(token=TOKEN, endpoint=ENDPOINT)

    # Submit training data to DataRobot for modelling:
    proj = dr.Project.create(
        modelling_data.toPandas(),
        project_name='SAMPLE')

    proj.set_target(
        target='targat_column',
        mode=dr.AUTOPILOT_MODE.QUICK,
        max_wait=3600,
        worker_count=-1
    )

    # Once the modeling autopilot is started, lock the interpreter to wait for autopilot to finish
    proj.wait_for_autopilot()
