import pathlib
from dataclasses import dataclass
from .esrf_nexus import ESRFNexus


@dataclass
class Filename:
    filename: str
    folder: str
    session_folder: str
    sample_name: str
    dataset_number: int

    def __str__(self):
        return str(self.filename)


def select_samplename(folder):
    if not folder.exists():
        raise ValueError(f"{folder} does not exists")
    samples = sorted(list(folder.glob("*")))
    samples = [s for s in samples if s.is_dir()]
    print("Available samples:")
    for i, s in enumerate(samples, 1):
        print(f"{i:2d} : {s}")
    ans = input("Select number ")
    sample = samples[int(ans) - 1]
    sample = sample.parts[-1]  # take only last sub folder
    return sample


def select_sampledataset(folder):
    if not folder.exists():
        raise ValueError(f"{folder} does not exists")
    datasets = sorted(list(folder.glob("*")))
    datasets = [s for s in datasets if s.is_dir()]
    if len(datasets) == 1:
        return 1
    print("Available dataset:")
    for i, s in enumerate(datasets, 1):
        print(f"{i:2d} : {s}")
    ans = input("Select number ")
    dataset = datasets[int(ans) - 1]
    dataset = dataset.parts[-1]  # take only last sub folder
    dataset = int(dataset.split("_")[-1])
    return dataset


def _get_sessions(data_folder):
    content = data_folder.glob("*")
    folders = [c for c in content if c.is_dir()]
    # sessions should be 8 digits
    sessions = [f for f in folders if f.parts[-1].isdigit() and len(f.parts[-1]) == 8]
    sessions = sorted(sessions)
    return sessions


def get_filename(
    experiment,
    sample_name=None,
    dataset_number=1,
    base_folder="/data/visitor/",
    beamline="id10",
    session_date="last",
):
    """Get filename assuming esrf data policy convention
       It looks for 
       base_folder
         └ experiment
             └ beamline
                 └ session
                     └ raw             ← This level might not exist
                         └ sample_name
                             └ {sample_name}_{dataset_number}
                                  └ {sample_name}_{dataset_number}.h5

    The script handles the lack of raw subfolder

    Parameters
    ----------
    experiment : str
        experimentname (for example sc5359)
    sample_name: None or str
        if None, interactive selection of sample_name
    dataset_number: None or int
        if None, interactive selection of dataset_number
    base_folder : str
    beamline: str
    session_date: str
        if 'last' uses last session_date subfolder
        if no session_date subfolder is present, it uses the main folder
    """

    experiment_folder = pathlib.Path(f"{base_folder}") / experiment / beamline

    if session_date == "last":
        sessions = _get_sessions(experiment_folder)
        if len(sessions) > 0:
            session_folder = sessions[-1]  # get last session by default
        else:
            session_folder = experiment_folder
    else:
        session_folder = base_folder / session_date

    # check is subfolder "raw" exists (should be present for experiments
    # starting in Jan 2023
    raw = session_folder / "raw"
    if raw.exists():
        session_folder = raw

    if sample_name is None:
        sample_name = select_samplename(session_folder)
    if dataset_number is None:
        dataset_number = select_sampledataset(session_folder / sample_name)

    name = f"{sample_name}_{dataset_number:04d}"
    fname = pathlib.Path(f"{session_folder}/{sample_name}/{name}/{name}.h5")
    return Filename(
        filename=fname,
        session_folder=session_folder,
        folder=fname.parent,
        sample_name=sample_name,
        dataset_number=dataset_number,
    )


def get_dataset(
    experiment,
    sample_name=None,
    dataset_number=1,
    base_folder="/data/visitor/",
    beamline="id10",
    session_date="last",
):
    """Get filename assuming esrf data policy convention

    Parameters
    ----------
    see doc string of get_filename
    """
    fname = get_filename(
        experiment,
        sample_name=sample_name,
        dataset_number=dataset_number,
        base_folder=base_folder,
        beamline=beamline,
        session_date=session_date,
    )

    return ESRFNexus(fname)
