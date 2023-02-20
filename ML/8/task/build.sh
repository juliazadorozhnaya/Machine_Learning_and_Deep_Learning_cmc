set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk

    pip3 install -qq \
        pytest==6.2.5 \
        scikit-image==0.18.3 \
        h5py==3.4.0 \
        catboost==0.26.1 \
        hyperopt==0.2.5 \
        ipywidgets==7.6.4 \
        keras==2.6.0 \
        lightgbm==3.2.1 \
        matplotlib==3.4.3 \
        numpy==1.21.2 \
        pandas==1.3.2 \
        plotly==5.3.1 \
        scipy==1.7.1 \
        seaborn==0.11.2 \
        scikit-learn==1.0.2 \
        torch==1.9.0 \
        torchvision==0.10.0 \
        tqdm==4.62.2 \
        umap-learn==0.5.1 \
        xgboost==1.4.2 \
        pep8==1.7.1
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"