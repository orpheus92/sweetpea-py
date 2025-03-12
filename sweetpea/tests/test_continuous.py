import operator as op
import pytest

from sweetpea._internal.primitive import Factor, DerivedLevel, WithinTrial, Transition, Window, SimpleLevel, ContinuousFactor
from sweetpea import synthesize_trials, RandomGen, MinimumTrials, CrossBlock
from sweetpea._internal.constraint import ConstinuousConstraint
import random
import numpy as np
from typing import cast
import math

from sweetpea._internal.sampling_strategy.sampling_continue import (
    UniformSampling, GaussianSampling, 
    ExponentialSampling, LogNormalSampling, CustomSampling
)



color = Factor("color", ["red", "blue", "green", "brown"])

def sample_continuous():
    return random.uniform(0.5, 1.5)
time_sample_function = ContinuousFactor("time_sample_function", [], sampling_function=CustomSampling(sample_continuous))

# Different sampling methods
time_uniform = ContinuousFactor("time_uniform", [], sampling_function=UniformSampling(0,10))
time_gaussian = ContinuousFactor("time_gaussian", [], sampling_function=GaussianSampling(0,1))
time_exponential = ContinuousFactor("time_exponential", [], sampling_function=ExponentialSampling(1))
time_lognormal = ContinuousFactor("time_lognormal", [], sampling_function=LogNormalSampling(0,1))

# Derived Factors
def difference(t1, t2):
    return t1-t2

difference_time = ContinuousFactor("difference_time", [
    time_uniform, time_gaussian], sampling_function=CustomSampling(difference))

difference_time1 = ContinuousFactor("difference_time1", [
    difference_time, time_exponential], sampling_function=CustomSampling(difference))

def color2time(color):
    if color == "red":
        return random.uniform(0, 1)
    elif color == "blue":
        return random.uniform(1, 2)
    elif color == "green":
        return random.uniform(2, 3)
    else:
        return random.uniform(3, 4)

color_time = ContinuousFactor("color_time", [
    color], sampling_function=CustomSampling(color2time))

# Constraints
def greater_than_2(a, b):
    return (a+b>2)
cc = ConstinuousConstraint([time_gaussian, time_exponential], greater_than_2)


# Factor Tests
def test_factor_type():
    assert isinstance(time_sample_function, ContinuousFactor) == True
    assert isinstance(time_uniform, ContinuousFactor) == True
    assert isinstance(time_gaussian, ContinuousFactor) == True
    assert isinstance(time_exponential, ContinuousFactor) == True
    assert isinstance(time_lognormal, ContinuousFactor) == True

def test_sampling_range():
    t1 = []
    t2 = []
    t3 = []
    for i in range(1000):
        result = time_sample_function.generate()
        result1 = time_uniform.generate()
        t1.append(time_gaussian.generate())
        t2.append(time_exponential.generate())
        t3.append(time_lognormal.generate())
        # Check if the value is between 0 and 1
        assert 0.5<= result <= 1.5
        assert 0<= result1 <= 10

    def normal_cdf(x, mean, std):
        """Calculates the CDF of a normal distribution using the error function (erf)."""
        z = (x - mean) / (std * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def kolmogorov_smirnov_test(samples, mean, std):
        """Performs the Kolmogorov-Smirnov test for normal distribution."""
        samples.sort()
        n = len(samples)
        max_diff = 0
        for i in range(n):
            cdf_val = normal_cdf(samples[i], mean, std)
            # Use midpoint for better alignment
            diff = abs(cdf_val - ((i + 0.5) / n))
            max_diff = max(max_diff, diff)
        return max_diff

        return max_diff

    def check_normal(samples, mean, std):
        """Checks if samples follow a normal distribution using the K-S test."""
        if not samples:
            return False, "No samples provided"
        ks_stat = kolmogorov_smirnov_test(samples, mean, std)
        # Critical value approximation for K-S test
        critical_value = 1.36 / math.sqrt(len(samples))
        result = ks_stat < critical_value
        message = f"K-S Statistic: {ks_stat:.5f}, Critical Value: {critical_value:.5f}, Mean: {mean:.5f}, Std Dev: {std:.5f}"
        return result, message

    result, message = check_normal(t1, 0, 1)
    assert result, f"The samples likely do not follow a normal distribution. {message}"

    def exponential_cdf(x, lambda_val):
        """Calculates the cumulative distribution function (CDF) of an exponential distribution."""
        return 1 - math.exp(-lambda_val * x)

    def kolmogorov_smirnov_test(samples, lambda_val):
        """Performs the Kolmogorov-Smirnov test to check for exponential distribution."""
        samples.sort()
        n = len(samples)
        max_diff = 0
        for i in range(n):
            cdf_val = exponential_cdf(samples[i], lambda_val)
            diff1 = abs(cdf_val - (i / n))
            diff2 = abs(cdf_val - ((i + 1) / n))
            max_diff = max(max_diff, diff1, diff2)
        return max_diff

    def check_exponential(samples, lambda_val):
        """Checks if samples follow an exponential distribution using the K-S test."""
        if not samples:
            return False, "No samples provided"
        mean = sum(samples) / len(samples)
        ks_stat = kolmogorov_smirnov_test(samples, lambda_val)
        # Critical value approximation for K-S test (large samples)
        critical_value = 1.36 / math.sqrt(len(samples))
        return ks_stat < critical_value, f"K-S Statistic: {ks_stat}, Critical Value: {critical_value}"

    rate = 1 
    is_exponential, result_string = check_exponential(t2, rate)
    assert is_exponential == True, f"The samples likely do not follow an exponential distribution. {result_string}"
    
    
    z975 = 1.96
    # Estimate the mean and std of the transformed normal data
    mu = np.mean(t3)
    sigma = np.std(t3, ddof=1)
    n = len(t3)

    # 95% Confidence interval for the normal distribution

    ci_lower = mu - z975 * (sigma / np.sqrt(n))
    ci_upper = mu + z975 * (sigma / np.sqrt(n))
    ci_lognorm_lower = np.exp(ci_lower)
    ci_lognorm_upper = np.exp(ci_upper)

    # Calculate the mean of the lognormal distribution (this is different from the log mean)
    mean_lognorm = np.exp(mu + (sigma**2 / 2))

    # Assertions: Check that the log-transformed mean is within the CI bounds
    assert ci_lower < mu < ci_upper, f"Mean for lognormal distribution {mu} is outside of the CI bounds ({ci_lower}, {ci_upper})"
    

def test_factor_validation():
    # This will use a default sampling function
    
    # Incorrect sampling method name
    with pytest.raises(TypeError):
        ContinuousFactor('response_time', [], sampling_function=UniformSampling(1))

    # Incorrect sampling method name
    with pytest.raises(TypeError):
        ContinuousFactor('response_time', [], sampling_function=ExponentialSampling(1,1))

    with pytest.raises(ValueError):
        ContinuousFactor('response_time', [], sampling_function=1)


def test_factor_get_level():
    assert time_sample_function.get_level("red") is None
    assert time_sample_function.get_level(1) is None
    assert time_sample_function.get_levels() == []
    assert time_sample_function.initial_levels == []
    assert difference_time.get_level(1) is None
    assert difference_time.get_levels() == [time_uniform, time_gaussian]
    assert difference_time.initial_levels == [time_uniform, time_gaussian]

def test_factor_has_complex_window():
	assert time_sample_function.has_complex_window == False    
	assert difference_time.has_complex_window == False

def test_factor_applies_to_trial():
    assert difference_time.applies_to_trial(1) == True
    assert difference_time.applies_to_trial(2) == True
    assert time_sample_function.applies_to_trial(1) == True
    assert time_sample_function.applies_to_trial(2) == True

    with pytest.raises(ValueError):
        time_sample_function.applies_to_trial(0)


# Block Tests
def test_block_creation():
    block = CrossBlock([color, time_sample_function], [color], [])

    with pytest.raises(RuntimeError):
        CrossBlock([color, time_sample_function], [color, time_sample_function], [])

    with pytest.raises(RuntimeError):
        CrossBlock([color, time_sample_function], [time_sample_function], [])

def test_has_factor():
    block = CrossBlock([color, time_sample_function], [color], [])
    assert block.has_factor(time_sample_function) == time_sample_function
    assert block.has_factor(difference_time) == cast(Factor, None)

def test_block_size():
    block = CrossBlock([color, time_sample_function], [color], [])
    assert block.crossing_size() == 4

# Trial Factor Dependency Tests
def test_trial_factor_dependence():
    design = [color, time_uniform, time_gaussian, time_exponential, difference_time, difference_time1]
    crossing = [color]
    block = CrossBlock(design, crossing, [MinimumTrials(20)])
    experiments  = synthesize_trials(block, 2)

    for ind in range(len(experiments)):
        time_uniform_sample = np.array(experiments[ind][time_uniform.name])
        time_gaussian_sample = np.array(experiments[ind][time_gaussian.name])
        time_exponential_sample = np.array(experiments[ind][time_exponential.name])
        difference_time_sample = np.array(experiments[ind][difference_time.name])
        difference_time1_sample = np.array(experiments[ind][difference_time1.name])
        # print(time_gaussian_sample, time_exponential_sample)
        assert np.array_equal(time_uniform_sample - time_gaussian_sample, difference_time_sample)
        assert np.array_equal(difference_time_sample - time_exponential_sample, difference_time1_sample)

    design = [color, color_time]
    crossing = [color]
    block = CrossBlock(design, crossing, [MinimumTrials(20)])
    experiments  = synthesize_trials(block, 2)

    for ind in range(len(experiments)):
        color_trial = experiments[ind][color.name]
        color_time_trial = experiments[ind][color_time.name]
        
        for i in range(len(color_trial)):
            if color_trial[i] == "red":
                assert 0 <= color_time_trial[i]<=1
            if color_trial[i] == "blue":
                assert 1 <= color_time_trial[i]<=2
            if color_trial[i] == "green":
                assert 2 <= color_time_trial[i]<=3
            if color_trial[i] == "brown":
                assert 3 <= color_time_trial[i]<=4

# Continuous Constraint Tests
def test_continuous_constraint():
    design = [color, time_gaussian, time_exponential]
    crossing = [color]
    block = CrossBlock(design, crossing, [cc])
    experiments  = synthesize_trials(block, 2)

    for ind in range(len(experiments)):
        time_gaussian_sample = np.array(experiments[ind][time_gaussian.name])
        time_exponential_sample = np.array(experiments[ind][time_exponential.name])
        sum_sample = time_gaussian_sample+time_exponential_sample
        assert np.all(sum_sample > 2)