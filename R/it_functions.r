decipherCurve_v2024_2 <- function(data_line, threshold_error = 4000, threshold_low = 10000, threshold_high = 30000, italiano = FALSE) {
  warning_flag <- 0
  # nomenclature des flags :
  # 1 = valeurs manquantes ou aberrantes dans la courbe
  # 2 = variation entre 4000 et 10000 (aberrantes)
  # 4 = nombre de délimiteurs dans la courbe différent du nombre annoncé
  # extrait des données depuis la ligne de la db
  if (italiano) {
    s <- data_line$curva
    fractions_count <- data_line$numfraz
    fractions_names <- unlist(data_line[ ,paste0("nome_", 1:fractions_count)])
    s_peaks <- data_line$coord_picco
  } else {
    s <- data_line$raw_curve
    fractions_count <- data_line$number_fractions
    fractions_names <- unlist(data_line[ ,paste0("fraction", 1:fractions_count, "_name")])
    s_peaks <- data_line$peaks_raw_coords
  }
  # extraction du nom des fractions
  # attention ! dans la db il n'y a que 6 colonnes pour les noms des fractions
  # si la ligne contient plus de 6 fractions, on ne peut pas extraire les noms !
  if (fractions_count > 6) {
    fractions_names <- rep('?', fractions_count)
  }
  # cassage du bloc hexadécimal en valeurs décimales
  s <- gsub("(.{4})", "\\1 ", s)
  s <- strsplit(s, " ")[[1]]
  s <- strtoi(s, 16L)
  # s'il ya des valeurs manquantes ou aberrantes, on les remplace par 0 en mettant un warning flag
  if (sum(is.na(s)) > 0) {
    # on remplace les valeurs manquantes par 0
    warning_flag <- bitwOr(warning_flag, 1)
    s[is.na(s)] <- 0
  }
  s <- rev(s) # ici, on inverse la courbe (gauche à droite vs droite à gauche)
  # tmps <- s[s < threshold_low]
  # ggplot(data.frame(x = 1:(length(tmps)), y = tmps), aes(x = x, y = y )) +
  #   geom_line() +
  #   theme_minimal()
  # Ici, on va ajouter une étape où l'on cherche s'il y a un saut de plus de 4000 à un endroit du spectre
  d <- diff(s)
  if (sum(abs(d) > threshold_error & abs(d) < threshold_low) > 0) {
    # Si oui, on va ajouter un flag de warning pour le signaler
    warning_flag <- bitwOr(warning_flag, 2)
    # Changement : on ne corrige plus les erreurs
    # Puis corriger les valeurs pour les mettre à 0
    # i <- which(d > threshold_error & d < threshold_low)
    # s[(i+1):(length(s)-1)] <- 0
    # s
  }
  # repérage des séparations de fractions
  delimitations <- which(s >= threshold_low) # on extrait les valeurs aberrantes qui sont des séparations de fractions/sous-fractions
  delimitations_values <- s[delimitations] # on extrait la valeur correspondant à chacune de ces délimitations, pour pouvoir séparer les normales des secondaires (sous-fractions)
  # on remplace les valeurs à l'endroit du fractionnement par la moyenne avant/après (ex: 2 35000 4 devient 2 3 4)
  # puisqu'il peut en théorie y avoir plusieurs délimitations d'affilée, où la courbe peut commencer/finir par une délim, on ne va pas
  # simplement prendre la "valeur juste avant" ou juste après
  values = s
  for (delim in delimitations) {
    replacement_value = NA
    # first point
    if ((delim == 1) || (length(setdiff(1:delim, delimitations)) == 0)) {
      # no real value before this delimitation => take first point non delim
      first_real_point = min(setdiff(1:length(s), delimitations))
      replacement_value = s[first_real_point]
    }
    # 2nd case: if delim == last // no real point after
    else if ((delim == length(s)) || (length(setdiff(delim:length(s), delimitations)) == 0)) {
      # no real value before this delimitation => take first point non delim
      last_real_point = min(setdiff(1:length(s), delimitations))
      replacement_value = s[last_real_point]
    } else {
      # TODO 3rd case: if delim is in the middle (take into account if another delim right after/before!)
      # take the closest value to the left and to the right, except values in delimitations
      prev_i = max(setdiff(1:delim, delimitations))
      post_i = min(setdiff(delim:length(s), delimitations))
      if ((post_i - prev_i) > 2) {
        warning_flag <- bitwOr(warning_flag, 8)  # warning: values are more than 2 steps away
      }
      replacement_value = round((s[prev_i] + s[post_i]) / 2)
    }
    if (is.na(replacement_value)) {
      stop("No value to replace delimiter")
    }
    values[delim] = replacement_value
  }
  # on sépare les fractions de sous-fractions (valeurs > 30000 ou entre 10000 et 30000)
  fractions <- delimitations[delimitations_values >= threshold_high]
  subfractions <- delimitations[delimitations_values < threshold_high]
  # on vérifie s'il y a bien le nombre attendu de fractions avant de renvoyer le tout
  # s'il on a 6 fractions, il doit y avoir 5 séparateurs + 1 au début + 1 à la fin soit 1 de plus que de fractions
  if (length(fractions) != (fractions_count+1)) {
    warning_flag <- bitwOr(warning_flag, 4)
  }
  # Enfin, on va récupérer les coordonnées des pics si présents
  s2 <- s_peaks
  if (grepl("^[0-9]+$", s2)) {
    s2 <- strsplit(gsub("(.{3})", "\\1 ", s2), " ")[[1]]
    peaks_coords <- as.numeric(s2)
    peaks_coords <- length(values) - peaks_coords
    # Dernière étape pour les pics : les coordonnées peuvent être décalées par le retrait des fractions et
    # des sous-fractions, il faut donc corriger ce problème :
    # peaks_coords <- peaks_coords + 1
    if (length(peaks_coords) <= 1) {
      peaks_coords <- numeric(0)
    }
    if (length(peaks_coords) > 0) {
      for(pi in 1:length(peaks_coords)) {
        peaks_coords[pi] <- peaks_coords[pi] + sum(c(fractions, subfractions) > peaks_coords[pi])
      }
    }
  } else {
    peaks_coords <- numeric(0)
  }

  # Puis on renvoie
  return(list(curve = values,
              fractions_names = fractions_names,
              fractions_coords = fractions,
              subfractions_coords = subfractions,
              peaks_coords = peaks_coords,
              alertflag = warning_flag))
}

drawProcessedSPE <- function(y, use_ggplot = TRUE) {
  require(ggplot2)
  if (use_ggplot) {
    gp <- ggplot()

    gp <- gp +
      # courbe de l'analyse
      geom_line(data = data.frame(x = c(1:length(y)), y = y), aes(x = x, y = y), size = 1)

    gp <- gp +
      theme_minimal()

    gp
  } else {
    plot(y~x, data.frame(x = c(1:length(y)), y = y), type = 'l')
  }
}

na_padding_end <- function(vect, total_length) {
  if (length(vect) == total_length) {
    return(vect)
  }
  if (length(vect) > total_length) {
    stop("Length of vector exceeds expected total length")
  }
  return(c(vect, rep(NA, total_length - length(vect))))
}

zero_padding_balanced <- function(vect, total_length) {
  if (length(vect) == total_length) {
    return(vect)
  }
  if (length(vect) > total_length) {
    stop("Length of vector exceeds expected total length")
  }
  padding_total = total_length - length(vect)
  padding_left = round(padding_total / 2)
  padding_right = padding_total - padding_left
  list(new_vect=c(rep(0, padding_left), vect, rep(0, padding_right)), padding_left=padding_left)
}

convert_to_spep_matrix <- function(db_full, normalize_01=TRUE, spep_extend_size=304, italiano=T) {
  require(progress)

  n_cols_spep = spep_extend_size
  if (is.na(n_cols_spep)) {
    n_cols_spep <- 300
  }
  n_cols_fractions = 10
  n_cols_peaks = 10
  n_cols_alertflag = 1
  n_cols_total = n_cols_spep + n_cols_fractions + n_cols_peaks + n_cols_alertflag
  # create our data matrix
  spe_matrix = matrix(data=NA, nrow=nrow(db_full), ncol=n_cols_total)  # total number of entries for one line
  # create progress bar
  pb <- progress_bar$new(total = nrow(db_full))
  for (i in 1:nrow(db_full)) {
    # extract SPEP data
    tmp <- decipherCurve_v2024_2(db_full[i, ], italiano=italiano)
    if (!is.na(spep_extend_size) && (spep_extend_size > 0)) {
      padded_curve <- zero_padding_balanced(tmp$curve, total_length = spep_extend_size)
      tmp$curve <- zero_padding_balanced(tmp$curve, total_length = spep_extend_size)$new_vect
      tmp$fractions_coords <- tmp$fractions_coords + padded_curve$padding_left
      tmp$peaks_coords <- tmp$peaks_coords + padded_curve$padding_left
    }
    if (normalize_01) {
      tmp$curve <- (tmp$curve - min(tmp$curve)) / (max(tmp$curve) - min(tmp$curve))
    }

    # fill data into a new line
    matrix_new_line <- c(na_padding_end(tmp$curve, n_cols_spep),
                         na_padding_end(tmp$fractions_coords, n_cols_fractions),  # 6 fractions + up to 3 fractions - 1 + 2 (first & last delimiters)
                         na_padding_end(tmp$peaks_coords, n_cols_peaks),  # up to 5 peaks (?)
                         tmp$alertflag)
    # save
    spe_matrix[i, ] <- matrix_new_line
    # update progress bar
    pb$tick()
  }

  # turn into data.frame, change column names, add identification number and save
  spe_df = data.frame(spe_matrix)
  # change column names
  colnames(spe_df) <- c(paste0("x", 1:n_cols_spep),
                        paste0("f", 1:n_cols_fractions),
                        paste0("p", 1:n_cols_peaks),
                        "alert_flag")
  # add id and other columns
  spe_df
}