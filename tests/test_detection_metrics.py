import pytest

from eval.detection_metrics import average_precision, counts, match_image, summarize, support_partition
from eval.visdrone import coverage, parse_annotations


def pred(identifier=0, box=(0, 0, 10, 10), score=.9, cls="person", support="Screening-Only"):
    return {"id": identifier, "box": box, "score": score, "class": cls, "support_label": support}


def annotation(targets=None, regions=None):
    return {"width": 100, "height": 100, "regions": regions or [], "targets": targets or []}


def gt(identifier=0, box=(0, 0, 10, 10), cls="person", ignore=False):
    return {"id": identifier, "box": box, "class": cls, "ignore": ignore}


def test_hand_calculated_point_metrics():
    m = counts(8, 2, 12)
    assert (m["tp"], m["fp"], m["fn"]) == (8, 2, 4)
    assert m["precision"] == .8
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(16 / 22)


def test_hand_calculated_101_point_ap():
    # TP, FP, TP -> precision envelope 1, 2/3, 2/3; recall .5,.5,1.
    items = [dict(pred(i, score=score), status=status) for i, (score, status) in enumerate([(.9, "TP"), (.8, "FP"), (.7, "TP")])]
    ap, curve = average_precision(items, 2)
    assert ap == pytest.approx((51 + 50 * (2 / 3)) / 101)
    assert len(curve) == 101
    assert curve[50]["precision"] == 1
    assert curve[51]["precision"] == pytest.approx(2 / 3)


def test_class_aware_one_to_one_confidence_order():
    results = match_image([pred(0, score=.8), pred(1, score=.9), pred(2, cls="car")], annotation([gt()]))
    assert [(p["id"], p["status"]) for p in results] == [(1, "TP"), (2, "FP"), (0, "FP")]


def test_iou_boundary_and_tie_break():
    a = annotation([gt(4), gt(2)])
    assert match_image([pred(box=(0, 0, 5, 10))], a)[0]["matched_gt"] == 2
    assert match_image([pred(box=(0, 0, 4.999, 10))], a)[0]["status"] == "FP"


def test_union_ignore_coverage_not_double_counted():
    assert coverage((0, 0, 10, 10), [(0, 0, 4, 10)] * 2) == .4
    assert coverage((0, 0, 10, 10), [(0, 0, 3, 10), (2, 0, 5, 10)]) == .5
    a = annotation([gt()], [(0, 0, 5, 10)])
    assert match_image([pred()], a)[0]["status"] == "ignored_region"


def test_valid_gt_before_same_class_score_zero_ignore():
    a = annotation([gt(), gt(1, ignore=True)])
    assert [p["status"] for p in match_image([pred(0), pred(1), pred(2)], a)] == ["TP", "ignored_score_zero", "ignored_score_zero"]
    assert match_image([pred(cls="car")], a)[0]["status"] == "FP"


def test_mapping_raw_categories_and_region_exclusion(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0,0,5,10,0,0,0,0\n0,0,10,10,1,1,0,0\n20,0,10,10,1,2,0,0\n40,0,10,10,1,4,0,0\n60,0,10,10,0,5,0,0\n80,0,10,10,0,1,0,0\n")
    a = parse_annotations(path, 100, 100)
    assert a["region_excluded_gt"] == 1
    assert [(g["class"], g["ignore"]) for g in a["targets"]] == [("person", False), ("car", False), ("person", True)]


@pytest.mark.parametrize("line", ["0,0,1,1,1,12,0,0", "0,0,0,1,1,1,0,0", "0,0,1,1,1,1,0", "nan,0,1,1,1,1,0,0"])
def test_bad_annotations_fail(tmp_path, line):
    path = tmp_path / "bad.txt"
    path.write_text(line)
    with pytest.raises(ValueError):
        parse_annotations(path, 100, 100)


def test_empty_and_absent_classes():
    result, _, _ = summarize({"a": []}, {"a": annotation([gt()])}, ranking=True)
    assert result["per_class"]["person"]["ap50"] == 0
    assert result["per_class"]["person"]["precision"] is None
    assert result["per_class"]["person"]["f1"] == 0
    assert result["per_class"]["car"]["ap50"] is None
    assert average_precision([], 0) == (None, [])


def test_perfect_predictions_have_ap_one_all_thresholds():
    result, _, _ = summarize({"a": [pred()]}, {"a": annotation([gt()])}, ranking=True)
    assert result["per_class"]["person"]["map50_95"] == 1


def test_fixed_partition_does_not_rematch_or_invent_fn():
    assignments = match_image([pred(0, support="Dual-Model"), pred(1, score=.8)], annotation([gt()]))
    result = support_partition(assignments)["combined"]
    assert result["Dual-Model"]["tp"] == 1
    assert result["Screening-Only"]["fp"] == 1
    assert result["precision_difference"] == 1
    assert "fn" not in result["Screening-Only"]


def test_zero_height_prediction_stops_rather_than_silently_dropping():
    with pytest.raises(ValueError, match="Invalid clipped box"):
        match_image([pred(box=(10, 100, 20, 100))], annotation())


def test_flagged_ranking_degenerate_is_fp_and_never_enters_geometry(monkeypatch):
    from eval import detection_metrics as metrics
    p = dict(pred(box=(0, 10, 10, 10)), geometry_status="degenerate_after_clipping")
    a = annotation([gt(ignore=True), gt(1)], [(0, 0, 100, 100)])
    monkeypatch.setattr(metrics, "coverage", lambda *_: pytest.fail("Degenerate must bypass ignore coverage"))
    monkeypatch.setattr(metrics, "clip", lambda *_: pytest.fail("Degenerate must bypass geometry calculations"))
    assignments = match_image([p], a)
    assert assignments[0]["status"] == "FP" and assignments[0]["box"] == p["box"]
    assert "matched_gt" not in assignments[0]
    summary, _, _ = summarize({"image": [p]}, {"image": a}, ranking=True)
    metric = summary["per_class"]["person"]
    assert (metric["prediction_count"], metric["fp"], metric["tp"], metric["fn"]) == (1, 1, 0, 1)
    assert metric["ignored_predictions"] == 0


def test_nonfinite_cannot_be_disguised_as_degenerate_fp():
    p = dict(pred(box=(0, 0, float("inf"), 0)), geometry_status="degenerate_after_clipping")
    with pytest.raises(ValueError, match="Nonfinite"):
        match_image([p], annotation())
