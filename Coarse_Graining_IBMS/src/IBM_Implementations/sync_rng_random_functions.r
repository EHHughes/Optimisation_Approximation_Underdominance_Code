library(SyncRNG)


sync_rng_choice <- function(arr, size, s, replace = FALSE, weights = NULL) {
# currently only implemented for 1D arrays
n <- length(arr)
out_arr <- NULL

if (!inherits(s, "SyncRNG")) {
  stop("Parameter 's' must be an instance of SyncRNG.")
}

if (!replace && size > n) {
  stop("Cannot take a larger sample than population when 'replace = FALSE'")
}

if (replace && !is.null(weights)) {
  stop("Sampling with weights and replacement not yet implemented!")
}

if (replace) {
  for (k in 1:size) {
    idx <- s$randi() %% length(arr) + 1
    out_arr <- c(out_arr, arr[idx])
}
} else {
  if (is.null(weights)) {
    out_arr <- s$shuffle(arr)[1:size]
  } else {
    weights <- weights / sum(weights)  # normalize

    chosen <- NULL

    for (k in 1:size) {
    cumulative_probabilities <- cumsum(weights)
    idx <- which.min(cumulative_probabilities < s$rand())
    chosen <- c(chosen, arr[idx])

    # Remove chosen element and its weight by position (use negative index, not !idx)
    arr <- arr[-idx]
    weights <- weights[-idx]

    # Renormalize remaining weights
    if (length(weights) > 0) {
    weights <- weights / sum(weights)
    }
    }

  out_arr <- chosen
  }
  }
  print
  return(out_arr)
}


sync_rng_binomial <- function(n, p, s) {
  n_successes <- 0

  for (i in 1:n) {
    n_successes <- n_successes + (s$rand() < p)
  }
  return(n_successes)
}


sync_rng_uniform <- function(
    lower_bounds,
    upper_bounds,
    size,
    s) {
  if (is.integer(size)) {
    out_array <- vector(length = size)
  } else {
    out_array <- vector(length = size[1]*size[2])
  }

  for (i in 1:length(out_array)) {
    out_array[i] <- s$rand()
  }

  if (!is.integer(size)) {
    out_array <- matrix(out_array, nrow = size[1], byrow = TRUE)
  }

  widths <- upper_bounds - lower_bounds

  out_array <- lower_bounds + widths * out_array

  return(out_array)
}