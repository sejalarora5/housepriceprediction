from regression_model.config.core import config
from regression_model.processing.features import TemporalVariableTransformer


def test_temporal_variable_transformer(sample_input_data):
    # Given
    transformer = TemporalVariableTransformer(
        variables=config.feature_config.temporal_vars,  # YearRemodAdd
        reference_variable=config.feature_config.ref_var,
    )
    assert sample_input_data["YearRemodAdd"].iat[0] == 1961

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["YearRemodAdd"].iat[0] == 49







# def test_other():
#     assert 2+2 == 4


# @pytest.fixture(params=["a", "b"])
# def demo_fixture(request):
#     print(request.param)
#     return request.param

# def test_this(demo_fixture):
#     print("Tanaa ")

# @pytest.mark.parametrize("a, b, sum", [(2,6,8), (1,2,3)])
# def test_addition(a, b, sum):
#     assert a + b == sum

# # @pytest.mark.skip
# # def test_my_patience():
# #     print("testing my patience")

# # @pytest.mark.xfail
# # def test_my_patience_twice():
# #     print("testing my patience twice")