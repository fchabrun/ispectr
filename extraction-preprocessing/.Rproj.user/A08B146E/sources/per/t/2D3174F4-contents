rm(list=ls())

# Objectives of this script:
# Prepare the data extracted from the Capillarys
# So we can load them into an interface
# Which will display the comments that the users wrote
# And an automatic SPECTR-peak localization analysis
# So we can choose wether to keep/modify those m-spikes

# EXTRACT_WITH_REF = F
# if (EXTRACT_WITH_REF) {
#   IS_TRACKS = c("Ref", "ELP", "IgG", "IgA", "IgM", "K", "L")
# } else {
#   IS_TRACKS = c("ELP", "IgG", "IgA", "IgM", "K", "L")
# }

source("it_functions.r")

library(openxlsx)
library(stringr)
library(tidyverse)
library(gridExtra)
library(progress)
library(data.table)
library(reticulate)
library(tools)
# library(rjson)
# library(jsonlite)
use_python("C:/Users/flori/anaconda3/envs/py311_dash")

db_file_path = "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/lemans_2025/EXPORT_DEIDENTIFIED_DATASET.csv"
db_data = as.data.frame(fread(db_file_path, sep=","))

print(paste0("File contains ", nrow(db_data), " raw entries"))

#check how many, etc.
db_data <- db_data %>%
  group_by(aaid) %>%
  mutate(n_spe = sum(program == "J")) %>%
  mutate(n_ref = sum(analysis_sequence == "Ref")) %>%
  mutate(n_elp = sum(analysis_sequence == "ELP")) %>%
  mutate(n_g = sum(analysis_sequence == "IgG")) %>%
  mutate(n_a = sum(analysis_sequence == "IgA")) %>%
  mutate(n_m = sum(analysis_sequence == "IgM")) %>%
  mutate(n_k = sum(analysis_sequence == "K")) %>%
  mutate(n_l = sum(analysis_sequence == "L")) %>%
  as.data.frame()

# also load "old" Le Mans data
old_lemans_x_path = "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/proc/lemans_2018/if_v1_x.npy"
old_lemans_yseg_path = "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/proc/lemans_2018/if_v1_y.npy"
old_lemans_y_path = "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/proc/lemans_2018/if_simple_y.csv"

np <- import("numpy")
old_lemans_x_matrix <- np$load(old_lemans_x_path)
old_lemans_y_matrix <- np$load(old_lemans_yseg_path)

print(paste0("Old Le Mans x dataset shape ", paste0(dim(old_lemans_x_matrix), collapse="x")))
print(paste0("Old Le Mans y dataset shape ", paste0(dim(old_lemans_y_matrix), collapse="x")))

# add y annot
old_lemans_annot <- read.csv(old_lemans_y_path)
print(paste0("Old Le Mans annotations dataset contains ", nrow(old_lemans_annot), " entries x ", ncol(old_lemans_annot), " columns"))
old_lemans_y_class <- old_lemans_annot[,1]

# finally add an md5 sum of the elp trace
# so we'll be able to easily compare and see if the trace is "known" in the old database
elp_trace_2_md5 <- function(elp_line) {
  elp_line_txt <- paste(round(elp_line, 3), collapse="")
  elp_line_md5 <- digest::digest(elp_line_txt, algo='md5')
  elp_line_md5
}

# columns_for_hash <- tail(head(paste0("ELP_", 1:304), 304-3), 304-6)  # remove 304 padding -> go back to 298
old_lemans_md5sum <- apply(old_lemans_x_matrix[,4:301,1], 1, elp_trace_2_md5)

# check if worked as expected
if (elp_trace_2_md5(elp_line = old_lemans_x_matrix[1, 4:301, 1]) != old_lemans_md5sum[1]) {
  stop("md5 not working properly?")
}

# check all unique
if (any(duplicated(old_lemans_md5sum))) {
  stop("Some md5 sums are duplicated!")
}

# FOR EACH SAMPLE ORGANIZE EVERYTHING
# WE WANT TO RETRIEVE AND ORGANIZE

input_json_directory <- "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/2025/lemans/preannotation/input_jsons"
prefilled_output_json_directory <- "C:/Users/flori/OneDrive - univ-angers.fr/Documents/Home/Research/SPECTR/ISPECTR/data/2025/lemans/preannotation/previous_2020_output_jsons"

std_comment <- function(coms) {
  coms <- coms[nchar(coms) > 3]
  coms <- gsub("[\r\n]+", " ", coms)
  coms <- paste(coms, collapse=" ")
  coms <- gsub("[ ]+", " ", coms)
  coms
}

postprocess_trace <- function(deciphered_curve) {
  if (!is.null(deciphered_curve)) {
    output_trace <- deciphered_curve$curve
    output_trace <- output_trace / max(output_trace)
    output_trace <- zero_padding_balanced(output_trace, 304)$new_vect
    return(output_trace)
  }
  return(rep(0, 304))
}

uniquify_value <- function(candidate_values) {
  candidate_values <- candidate_values[!is.na(candidate_values)]
  candidate_values <- candidate_values[candidate_values != "\\N"]
  if (length(unique(candidate_values)) == 1) {
    return(candidate_values[1])
  }
  return(NA)
}

unique_aaids <- unique(db_data$aaid)
pb = txtProgressBar(min = 0, max = length(unique_aaids), initial = 0)
skipped_list = c()
for (iter_i in 1:length(unique_aaids)) {
  iter_aaid = unique_aaids[iter_i]
  aaid_subset <- db_data[db_data$aaid == iter_aaid, ]
  
  # 1) CHECK IF EXACTLY 1 ELP, G, A, M, K, L
  # EXTRACT G A M K L ELP TRACES
  # EXTRACT IF PRESENT REF AND SPEP traces - skip if not present
  # DECIPHER CURVES
  # EXTRACT TEXT COMMENTS
  # STORE EVERYTHING INTO A SINGLE JSON FILE
  
  required_traces_are_present <- all(aaid_subset$n_elp == 1) &
    all(aaid_subset$n_g == 1) &
    all(aaid_subset$n_a == 1) &
    all(aaid_subset$n_m == 1) &
    all(aaid_subset$n_k == 1) &
    all(aaid_subset$n_l == 1)
  
  # also do other checks
  required_traces_are_consistent <- ((nchar(aaid_subset[aaid_subset$analysis_sequence == "ELP", ]$raw_curve) == 1200) &
                                       (nchar(aaid_subset[aaid_subset$analysis_sequence == "IgG", ]$raw_curve) == 1200) &
                                       (nchar(aaid_subset[aaid_subset$analysis_sequence == "IgA", ]$raw_curve) == 1200) &
                                       (nchar(aaid_subset[aaid_subset$analysis_sequence == "IgM", ]$raw_curve) == 1200) &
                                       (nchar(aaid_subset[aaid_subset$analysis_sequence == "K", ]$raw_curve) == 1200) &
                                       (nchar(aaid_subset[aaid_subset$analysis_sequence == "L", ]$raw_curve) == 1200))
  
  if (required_traces_are_present & required_traces_are_consistent) {
    exactly_1_spe_trace_is_present <- all(aaid_subset$n_spe == 1)
    exactly_1_ref_trace_is_present <- all(aaid_subset$n_ref == 1)
    
    at_least_1_short_comment <- any(nchar(aaid_subset$comment_1) >= 3)
    at_least_1_long_comment <- any(nchar(aaid_subset$comment_long) >= 3)
    
    # extract mandatory curves
    elp_deciphered <- decipherCurve_v2024_2(aaid_subset[aaid_subset$analysis_sequence == "ELP", ])
    g_deciphered <- decipherCurve_v2024_2(aaid_subset[aaid_subset$analysis_sequence == "IgG", ])
    a_deciphered <- decipherCurve_v2024_2(aaid_subset[aaid_subset$analysis_sequence == "IgA", ])
    m_deciphered <- decipherCurve_v2024_2(aaid_subset[aaid_subset$analysis_sequence == "IgM", ])
    k_deciphered <- decipherCurve_v2024_2(aaid_subset[aaid_subset$analysis_sequence == "K", ])
    l_deciphered <- decipherCurve_v2024_2(aaid_subset[aaid_subset$analysis_sequence == "L", ])
    
    # extract accessory traces
    ref_deciphered <- NULL
    if (exactly_1_ref_trace_is_present) {
      if (nchar(aaid_subset[aaid_subset$analysis_sequence == "Ref", ]$raw_curve) == 1200) {  # check the trace seems legit
        ref_deciphered <- decipherCurve_v2024_2(aaid_subset[aaid_subset$analysis_sequence == "Ref", ])
      } else {
        exactly_1_ref_trace_is_present <- F
      }
    }
    
    spe_deciphered <- NULL
    if (exactly_1_spe_trace_is_present) {
      if (nchar(aaid_subset[aaid_subset$program == "J", ]$raw_curve) == 1200) {  # check the trace seems legit
        spe_deciphered <- decipherCurve_v2024_2(aaid_subset[aaid_subset$program == "J", ])
      } else {
        exactly_1_spe_trace_is_present <- F
      }
    }
    
    # post-process traces
    elp_trace <- postprocess_trace(elp_deciphered)
    g_trace <- postprocess_trace(g_deciphered)
    a_trace <- postprocess_trace(a_deciphered)
    m_trace <- postprocess_trace(m_deciphered)
    k_trace <- postprocess_trace(k_deciphered)
    l_trace <- postprocess_trace(l_deciphered)
    
    if (exactly_1_ref_trace_is_present) {
      ref_trace <- postprocess_trace(ref_deciphered)
    } else {
      ref_trace <- NULL
    }
    
    spe_trace <- NULL
    spe_fractions <- NULL
    spe_peak_data <- NULL
    if (exactly_1_spe_trace_is_present) {
      # spe_trace <- postprocess_trace(spe_deciphered)
      # store spep trace
      spe_trace <- spe_deciphered$curve / max(spe_deciphered$curve)
      
      if ((length(spe_deciphered$fractions_coords) >= 5) & (length(spe_deciphered$fractions_coords) <= 7)) {
        # store spep fractions
        spe_fractions <- list(coords = spe_deciphered$fractions_coords,
                              names = spe_deciphered$fractions_names)
      }
      
      if (length(spe_deciphered$peaks_coords) > 0) {  # there are peaks
        if ((length(spe_deciphered$peaks_coords) %% 2) == 0) {  # even number of coords
          if (all(diff(spe_deciphered$peaks_coords) > 0)) {  # increasing positions
            # store spep peaks
            spe_peak_data <- spe_deciphered$peaks_coords
          }
        }
      }
    }
    
    # extract short and long comments
    short_comments <- std_comment(aaid_subset$comment_1)
    long_comments <- std_comment(aaid_subset$comment_long)
    # extract other important information
    age <- uniquify_value(aaid_subset$age_sampling)
    sex <- uniquify_value(aaid_subset$sex)
    tp <- as.numeric(uniquify_value(aaid_subset$total_protein))
    
    # store everything into a JSON
    sample_json_data <- list(paid = aaid_subset$paid[1],
                             aaid = aaid_subset$aaid[1],
                             age = age,
                             sex = sex,
                             total_protein = tp,
                             short_comments = short_comments,
                             long_comments = long_comments,
                             traces=list(ELP = list(exists=T, data=elp_trace),
                                         IgG = list(exists=T, data=g_trace),
                                         IgA = list(exists=T, data=a_trace),
                                         IgM = list(exists=T, data=m_trace),
                                         K = list(exists=T, data=k_trace),
                                         L = list(exists=T, data=l_trace),
                                         Ref = list(exists=exactly_1_ref_trace_is_present, data=ref_trace),
                                         SPE = list(exists=exactly_1_spe_trace_is_present, data=spe_trace, fractions=spe_fractions, peaks=spe_peak_data)))
    
    # save at desired location
    write(rjson::toJSON(sample_json_data, indent=2), file.path(input_json_directory, paste0(iter_aaid, ".json")))

    # CHECK FOR A MATCH and if yes -> pre-fill in the output file (will be stored in a separate folder: previous_2020_output_jsons)
    # in the past we used to "remove delimiters" while we are now replacing them with closest non-delimiter value
    # so "old" traces should be 298-long while new will be 300 points in all cases
    # thus, to "mimick" the old version of the trace, we just have to remove first and last values for ELP/GAMKL
    # since there should not be any other delimiter
    old_trace <- tail(head(elp_trace, 304-3), 304-6)
    # get the trace hash
    old_trace_md5 <- elp_trace_2_md5(old_trace)
    # compare to list of known md5 (old traces)
    if (old_trace_md5 %in% old_lemans_md5sum) {
      old_dataset_position <- which(old_lemans_md5sum == old_trace_md5)
      
      sample_output_json_data <- list(paid = aaid_subset$paid[1],
                                      aaid = aaid_subset$aaid[1],
                                      groundtruth_class=old_lemans_y_class[old_dataset_position],
                                      groundtruth_maps=list(IgG = old_lemans_y_matrix[old_dataset_position, , 1],
                                                            IgA = old_lemans_y_matrix[old_dataset_position, , 2],
                                                            IgM = old_lemans_y_matrix[old_dataset_position, , 3],
                                                            K = old_lemans_y_matrix[old_dataset_position, , 4],
                                                            L = old_lemans_y_matrix[old_dataset_position, , 5]))
      
      # save at desired location
      write(rjson::toJSON(sample_output_json_data, indent=2), file.path(prefilled_output_json_directory, paste0(iter_aaid, ".json")))
    }
  } else {
    skipped_list <- c(skipped_list, iter_aaid)
  }
  setTxtProgressBar(pb, iter_i)
}
close(pb)

print(paste0("Skipped ", length(skipped_list), " samples due to missing data"))

skipped_list
# [1] "AN_777c82fdc4fdc04b22b61fac1674a5cf" "AN_82dd62986f7515ff941643de50f73ba8"
# [3] "AN_fcf2327d2eddd0949d84fec93534f371" "AN_56b1c1a5e3cb9cb87b6857b306ca440f"
# [5] "AN_bde321431cece3511bb98483b9a8bfc7" "AN_160f813df159c72ee69c437577634058"
# [7] "AN_803dc4f6cc28d507ae93dd2021f7d24f" "AN_ec1b06bd5f19e695489965da43eb6959"
# [9] "AN_b15ae1a8db8c65a8c9fe3a9abea3901b" "AN_39314c5e119e613f683447403d701333"
# [11] "AN_ea5652ec3b0379f9eeb5ab9c99f9f5b9" "AN_3773e4e5febcb3d25bc543a043a758eb"
# [13] "AN_91e3def89936c26bd0d5221ab437c7ab" "AN_e849d61106d380ed95b42c3e6bc74ff1"
# [15] "AN_45232ed61ece94d000aedb077ef4ef71" "AN_f045b2a87366e579a40177580661f357"
# [17] "AN_abdabf5eecfd045cc13bf8e551c2c24e" "AN_4a7c29665a248c9cddb0af4a7b30e287"

# 5846 samples successfully extracted (seemingly)
# including 1724 for which we already had annotations
# we should have had 1803 but that's already nice (79 unmatched...? maybe removed in this updated version?)






























































