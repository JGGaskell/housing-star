import housing_star as hs


def test_public_api_is_importable():
    assert all(hasattr(hs, name) for name in hs.__all__)
    assert isinstance(hs.__version__, str)


def test_end_to_end_on_synthetic_data(ds):
    tuned = hs.tune(hs.StarModel, ds, {"p": [1, 2]}, hs.WalkForwardCV(n_folds=2, min_train=8))
    models = {"STAR": hs.StarModel(**tuned.best_params), "AR(1)": hs.ArModel(p=1)}
    report = hs.evaluate(ds, models)
    assert report.height == 2
    model = models["STAR"].fit(ds.train_only())
    scenario = hs.Scenario(
        "PC0",
        2,
        (hs.Perturbation("p", hs.NodeSelector(nth_closest=1), (hs.PerturbationStep(1, 5),)),),
    )
    assert hs.simulate(model, ds.train_only(), scenario).horizon == 2
