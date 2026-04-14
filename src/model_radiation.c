
/**********************************************************/
/*** all you need to make a polarized radiative transfer **/
/***** used in ipole to evolve complex tensor N ***********/
/***** along with standard evolution for I scalar *********/
/**********************************************************/
/**** written by Monika Moscibrodzka on 09 July 2014 ******/
/************ @ Eindhoven Airport *************************/
/************  last update: 9 May 2017   ******************/
/****************and then rewritten by C.Gammie ***********/
/**********************************************************/

#include "model_radiation.h"

#include "model.h"
#include "radiation.h"
#include "coordinates.h"
#include "geometry.h"
#include "grid.h"
#include "debug_tools.h"
#include "par.h"
#include "decs.h"

// Symphony
#include "fits.h"
#include "params.h"
#include "bremss_fits.h"

#include <omp.h>
#include <gsl/gsl_sf_bessel.h>

// Emission prescriptions, described below
#define E_THERMAL        1
#define E_KAPPA          2
#define E_POWERLAW       3
#define E_DEXTER_THERMAL 4
#define E_LEUNG          5
#define E_CUSTOM        10
// Rotation
#define ROT_OLD         11
#define ROT_PIECEWISE   12
#define ROT_SHCHERBAKOV 13
// Debugging/internal
#define E_UNPOL         15

#define RAD_OVERFLOW 1e100

// Local functions for declaring different kappa/powerlaw distributions
void get_model_powerlaw_vals(double X[NDIM], double *p, double *n,
                             double *gamma_min, double *gamma_max, double *gamma_cut);
void get_model_kappa(double X[NDIM], double *kappa, double *kappa_width);

/* Get coeffs from a specific distribution */
void jar_calc_dist(int dist, int pol, double X[NDIM], double Kcon[NDIM],
    double *jI, double *jQ, double *jU, double *jV,
    double *aI, double *aQ, double *aU, double *aV,
    double *rQ, double *rU, double *rV);

/**
 * Optionally load radiation model parameters
 */
static double model_kappa = 3.5;
static double powerlaw_gamma_cut = 1e10;
static double powerlaw_gamma_min = 1e2;
static double powerlaw_gamma_max = 1e5;
static double powerlaw_p = 3.25;
static double powerlaw_eta = 0.02;
static int variable_kappa = 0;
static double variable_kappa_min = 3.1;
static double variable_kappa_interp_start = 1e20;
static double variable_kappa_max = 7.0;
static double max_pol_frac_e = 0.99;
static double max_pol_frac_a = 0.99;
static int do_bremss = 0;
static int bremss_type = 2;

void try_set_radiation_parameter(const char *word, const char *value)
{
  set_by_word_val(word, value, "kappa", &model_kappa, TYPE_DBL);
  set_by_word_val(word, value, "variable_kappa", &variable_kappa, TYPE_INT);
  set_by_word_val(word, value, "variable_kappa_min", &variable_kappa_min, TYPE_DBL);
  set_by_word_val(word, value, "variable_kappa_interp_start", &variable_kappa_interp_start, TYPE_DBL);
  set_by_word_val(word, value, "variable_kappa_max", &variable_kappa_max, TYPE_DBL);

  set_by_word_val(word, value, "powerlaw_gamma_cut", &powerlaw_gamma_cut, TYPE_DBL);
  set_by_word_val(word, value, "powerlaw_gamma_min", &powerlaw_gamma_min, TYPE_DBL);
  set_by_word_val(word, value, "powerlaw_gamma_max", &powerlaw_gamma_max, TYPE_DBL);
  set_by_word_val(word, value, "powerlaw_p", &powerlaw_p, TYPE_DBL);
  set_by_word_val(word, value, "powerlaw_eta", &powerlaw_eta, TYPE_DBL);

  set_by_word_val(word, value, "bremss", &do_bremss, TYPE_INT);
  set_by_word_val(word, value, "bremss_type", &bremss_type, TYPE_INT);

  set_by_word_val(word, value, "max_pol_frac_e", &powerlaw_p, TYPE_DBL);
  set_by_word_val(word, value, "max_pol_frac_a", &powerlaw_p, TYPE_DBL);
}

/**
 * Get polarized emission, absorption, and rotation coefficients
 * 
 * This is a wrapper to jar_calc_dist, see implementation there
 * Also checks for NaN coefficients when built with DEBUG, for quicker debugging
 */
void jar_calc(double X[NDIM], double Kcon[NDIM],
    double *jI, double *jQ, double *jU, double *jV,
    double *aI, double *aQ, double *aU, double *aV,
    double *rQ, double *rU, double *rV, Params *params)
{
  jar_calc_dist(params->emission_type, 1, X, Kcon, jI, jQ, jU, jV, aI, aQ, aU, aV, rQ, rU, rV);

  // This wrapper can be used to call jar_calc_dist differently in e.g. funnel vs jet, or
  // depending on local fluid parameters, whatever
}

/**
 * Get the emission and absorption coefficients 
 * This is a wrapper to jar_calc_dist, see implementation there
 */
void get_jkinv(double X[NDIM], double Kcon[NDIM], double *jI, double *aI, Params *params)
{
  double jQ, jU, jV, aQ, aU, aV, rQ, rU, rV;
  jar_calc_dist(params->emission_type, 0, X, Kcon, jI, &jQ, &jU, &jV, aI, &aQ, &aU, &aV, &rQ, &rU, &rV);
}

/**
 * Get the invariant plasma emissivities, absorptivities, rotativities in tetrad frame
 * Calls the appropriate fitting functions from the code in symphony/,
 * and ensures that the results obey basic consistency
 *
 * The dist argument controls fitting functions/distribution:
 * 1. Thermal (Pandya+ 2016)
 * 2. Kappa (Pandya+ 2016, no rotativities)
 * 3. Power-law (Pandya+ 2016, rhoQ/rhoV Marszewski+ 2021)
 * 4. Thermal (Dexter 2016)
 * Thermal rhoQ taken from Dexter 2016, rhoV from Shcherbakov 2008 (rhoU == 0)
 *
 * To be implemented?
 * 5. Power-law (Dexter 2016)
 * 
 * Testing distributions:
 * 10. Pass through model-determined values -- used in most analytic models
 * 
 * 
 */
void jar_calc_dist(int dist, int pol, double X[NDIM], double Kcon[NDIM],
    double *jI, double *jQ, double *jU, double *jV,
    double *aI, double *aQ, double *aU, double *aV,
    double *rQ, double *rU, double *rV)
{
  int is_shock = get_model_kel(X);
  double n_nth = get_model_unth(X);
  double p_nth = get_model_p(X);
  double gamma_min_nth = get_model_gamma_min(X);
  double Ne = get_model_ne(X);
  int add_nonthermal = 0;
  struct parameters paramsM;
  double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
  double nu, nusq, theta;

  if (Ne <= 0. && !is_shock) {
    *jI = 0.0; *jQ = 0.0; *jU = 0.0; *jV = 0.0;
    *aI = 0.0; *aQ = 0.0; *aU = 0.0; *aV = 0.0;
    *rQ = 0.0; *rU = 0.0; *rV = 0.0;
    return;
  }

  if (dist == E_CUSTOM) {
    get_model_jar(X, Kcon, jI, jQ, jU, jV, aI, aQ, aU, aV, rQ, rU, rV);
    return;
  }

  setConstParams(&paramsM);
  get_model_fourv(X, Kcon, Ucon, Ucov, Bcon, Bcov);
  nu = get_fluid_nu(Kcon, Ucov);
  nusq = nu * nu;
  theta = get_bk_angle(X, Kcon, Ucov, Bcon, Bcov);

  paramsM.electron_density = Ne;
  paramsM.nu = nu;
  paramsM.observer_angle = theta;
  paramsM.magnetic_field = get_model_b(X);

  if (is_shock == 1) {
    static int hit_count = 0;
    if (hit_count++ % 100000 == 0) {
      printf("DEBUG: Ray hit shock! n_nth(UNTH)=%g Ne(thermal)=%g ratio=%.3e p=%g\n",
          n_nth, Ne, (Ne > 0.0) ? n_nth / Ne : -1.0, p_nth);
      printf("DEBUG: gamma_min=%g gamma_max=%g\n", gamma_min_nth, powerlaw_gamma_max);
    }
    paramsM.distribution = paramsM.MAXWELL_JUETTNER;
    paramsM.electron_density = Ne;
    paramsM.theta_e = get_model_thetae(X);
    add_nonthermal = (n_nth > 0.0 && p_nth > 1.0 && gamma_min_nth >= 1.0);
  } else {
    switch (dist) {
    case E_KAPPA:
      paramsM.distribution = paramsM.KAPPA_DIST;
      paramsM.dexter_fit = 1;
      paramsM.kappa_interp_begin = fmin(variable_kappa_interp_start, variable_kappa_max);
      paramsM.kappa_interp_end = variable_kappa_max;
      paramsM.theta_e = get_model_thetae(X);
      get_model_kappa(X, &(paramsM.kappa), &(paramsM.kappa_width));
      break;
    case E_POWERLAW:
      paramsM.distribution = paramsM.POWER_LAW;
      get_model_powerlaw_vals(X, &(paramsM.power_law_p), &(paramsM.electron_density),
                              &(paramsM.gamma_min), &(paramsM.gamma_max), &(paramsM.gamma_cutoff));
      break;
    case E_THERMAL:
      paramsM.distribution = paramsM.MAXWELL_JUETTNER;
      paramsM.theta_e = get_model_thetae(X);
      break;
    case E_DEXTER_THERMAL:
    default:
      paramsM.dexter_fit = 1;
      paramsM.distribution = paramsM.MAXWELL_JUETTNER;
      paramsM.theta_e = get_model_thetae(X);
      break;
    }
  }

  *jI = 0.0; *jQ = 0.0; *jU = 0.0; *jV = 0.0;
  *aI = 0.0; *aQ = 0.0; *aU = 0.0; *aV = 0.0;
  *rQ = 0.0; *rU = 0.0; *rV = 0.0;

  if (theta <= 0.0 || theta >= M_PI) {
    if (pol && !(dist == E_UNPOL)) {
      *rV = rho_nu_fit(&paramsM, paramsM.STOKES_V) * nu;
    }
    return;
  }

  if (!pol || dist == E_UNPOL) {
    if (paramsM.distribution == paramsM.MAXWELL_JUETTNER) {
      double Bnuinv;
      paramsM.dexter_fit = 2;
      *jI = j_nu_fit(&paramsM, paramsM.STOKES_I);
      if (do_bremss) *jI += bremss_I(&paramsM, bremss_type);
      *jI /= nusq;
      Bnuinv = Bnu_inv(nu, paramsM.theta_e);
      *aI = (Bnuinv > 0.0) ? (*jI / Bnuinv) : 0.0;
    } else {
      paramsM.dexter_fit = 2;
      *jI = j_nu_fit(&paramsM, paramsM.STOKES_I) / nusq;
      *aI = alpha_nu_fit(&paramsM, paramsM.STOKES_I) * nu;
    }

    if (add_nonthermal) {
      struct parameters paramsPL = paramsM;
      paramsPL.distribution = paramsPL.POWER_LAW;
      paramsPL.electron_density = n_nth;
      paramsPL.power_law_p = p_nth;
      paramsPL.gamma_min = gamma_min_nth;
      paramsPL.gamma_max = powerlaw_gamma_max;
      paramsPL.gamma_cutoff = powerlaw_gamma_cut;
      paramsPL.dexter_fit = 2;
      *jI += j_nu_fit(&paramsPL, paramsPL.STOKES_I) / nusq;
      *aI += alpha_nu_fit(&paramsPL, paramsPL.STOKES_I) * nu;
    }
  } else {
    double jP, aP, pol_frac_e, pol_frac_a;

    *jI = j_nu_fit(&paramsM, paramsM.STOKES_I);
    if (paramsM.distribution == paramsM.MAXWELL_JUETTNER && do_bremss)
      *jI += bremss_I(&paramsM, bremss_type);
    *jI /= nusq;
    *jQ = -j_nu_fit(&paramsM, paramsM.STOKES_Q) / nusq;
    *jU = -j_nu_fit(&paramsM, paramsM.STOKES_U) / nusq;
    *jV = j_nu_fit(&paramsM, paramsM.STOKES_V) / nusq;

    if (paramsM.distribution == paramsM.MAXWELL_JUETTNER) {
      double Bnuinv = Bnu_inv(nu, paramsM.theta_e);
      if (Bnuinv > 0.0) {
        *aI = *jI / Bnuinv;
        *aQ = *jQ / Bnuinv;
        *aU = *jU / Bnuinv;
        *aV = *jV / Bnuinv;
      }
    } else {
      *aI = alpha_nu_fit(&paramsM, paramsM.STOKES_I) * nu;
      *aQ = -alpha_nu_fit(&paramsM, paramsM.STOKES_Q) * nu;
      *aU = -alpha_nu_fit(&paramsM, paramsM.STOKES_U) * nu;
      *aV = alpha_nu_fit(&paramsM, paramsM.STOKES_V) * nu;
    }

    if (paramsM.distribution == paramsM.POWER_LAW) {
      *rQ = 0.0;
      *rU = 0.0;
      *rV = 0.0;
    } else {
      paramsM.dexter_fit = 0;
      *rQ = rho_nu_fit(&paramsM, paramsM.STOKES_Q) * nu;
      *rU = rho_nu_fit(&paramsM, paramsM.STOKES_U) * nu;
      *rV = rho_nu_fit(&paramsM, paramsM.STOKES_V) * nu;
    }

    if (add_nonthermal) {
      struct parameters paramsPL = paramsM;
      paramsPL.distribution = paramsPL.POWER_LAW;
      paramsPL.electron_density = n_nth;
      paramsPL.power_law_p = p_nth;
      paramsPL.gamma_min = gamma_min_nth;
      paramsPL.gamma_max = powerlaw_gamma_max;
      paramsPL.gamma_cutoff = powerlaw_gamma_cut;
      paramsPL.dexter_fit = 2;

      *jI += j_nu_fit(&paramsPL, paramsPL.STOKES_I) / nusq;
      *jQ += -j_nu_fit(&paramsPL, paramsPL.STOKES_Q) / nusq;
      *jU += -j_nu_fit(&paramsPL, paramsPL.STOKES_U) / nusq;
      *jV += j_nu_fit(&paramsPL, paramsPL.STOKES_V) / nusq;
      *aI += alpha_nu_fit(&paramsPL, paramsPL.STOKES_I) * nu;
      *aQ += -alpha_nu_fit(&paramsPL, paramsPL.STOKES_Q) * nu;
      *aU += -alpha_nu_fit(&paramsPL, paramsPL.STOKES_U) * nu;
      *aV += alpha_nu_fit(&paramsPL, paramsPL.STOKES_V) * nu;
    }

    jP = sqrt(*jQ * *jQ + *jU * *jU + *jV * *jV);
    if (jP > 0.0 && *jI < jP/max_pol_frac_e) {
      pol_frac_e = *jI / jP * max_pol_frac_e;
      *jQ *= pol_frac_e;
      *jU *= pol_frac_e;
      *jV *= pol_frac_e;
    }

    aP = sqrt(*aQ * *aQ + *aU * *aU + *aV * *aV);
    if (aP > 0.0 && *aI < aP/max_pol_frac_a) {
      pol_frac_a = *aI / aP * max_pol_frac_a;
      *aQ *= pol_frac_a;
      *aU *= pol_frac_a;
      *aV *= pol_frac_a;
    }
  }

#if DEBUG
  // Check for NaN coefficients
  if (isnan(*jI) || *jI > RAD_OVERFLOW || *jI < -RAD_OVERFLOW ||
      isnan(*jQ) || *jQ > RAD_OVERFLOW || *jQ < -RAD_OVERFLOW ||
      isnan(*jU) || *jU > RAD_OVERFLOW || *jU < -RAD_OVERFLOW ||
      isnan(*jV) || *jV > RAD_OVERFLOW || *jV < -RAD_OVERFLOW ||
      isnan(*aI) || *aI > RAD_OVERFLOW || *aI < -RAD_OVERFLOW ||
      isnan(*aQ) || *aQ > RAD_OVERFLOW || *aQ < -RAD_OVERFLOW ||
      isnan(*aU) || *aU > RAD_OVERFLOW || *aU < -RAD_OVERFLOW ||
      isnan(*aV) || *aV > RAD_OVERFLOW || *aV < -RAD_OVERFLOW ||
      isnan(*rQ) || *rQ > RAD_OVERFLOW || *rQ < -RAD_OVERFLOW ||
      isnan(*rU) || *rU > RAD_OVERFLOW || *rU < -RAD_OVERFLOW ||
      isnan(*rV) || *rV > RAD_OVERFLOW || *rV < -RAD_OVERFLOW) {
#pragma omp critical
    {
      fprintf(stderr, "\nNAN in emissivities!\n");
      fprintf(stderr, "j = %g %g %g %g alpha = %g %g %g %g rho = %g %g %g\n", *jI, *jQ, *jU, *jV, *aI, *aQ, *aU, *aV, *rQ, *rU, *rV);
      fprintf(stderr, "nu = %g Ne = %g Thetae = %g B = %g theta = %g kappa = %g kappa_width = %g\n",
              paramsM.nu, paramsM.electron_density, paramsM.theta_e, paramsM.magnetic_field, paramsM.observer_angle,
              paramsM.kappa, paramsM.kappa_width);
      // Powerlaw?
      exit(-1);
    }
  }
#endif
}

// SUPPORTING FUNCTIONS

/*
 * Get values for the powerlaw distribution:
 * power & cutoffs from parameters, plus number density of nonthermals
 */
void get_model_powerlaw_vals(double X[NDIM], double *p, double *n,
                             double *gamma_min, double *gamma_max, double *gamma_cut)
{
  *gamma_min = powerlaw_gamma_min;
  *gamma_max = powerlaw_gamma_max;
  *gamma_cut = powerlaw_gamma_cut;
  *p = powerlaw_p;

  double b = get_model_b(X);
  double u_nth = powerlaw_eta*b*b/2;
  // Number density of nonthermals
  *n = u_nth * (*p - 2)/(*p - 1) * 1/(ME * CL*CL * *gamma_min);
}

/*
 * Get values kappa, kappa_width for the kappa distribution emissivities
 * Optionally variable over the domain based on sigma and beta
 */
void get_model_kappa(double X[NDIM], double *kappa, double *kappa_width) {

  if (variable_kappa) {
    double sigma = get_model_sigma(X);
    double beta = get_model_beta(X);

    *kappa = 2.8 + 0.7/sqrt(sigma) + 3.7 * (1.0/pow(sigma, 0.19)) * tanh(23.4 * pow(sigma, 0.26) * beta);

    if (isnan(*kappa)) {
      double B = get_model_b(X);
      fprintf(stderr, "kappa, sigma, beta, B: %g %g %g %g \n", *kappa, sigma, beta, B);
    }
    // Lower limit for kappa.  Upper limit switches away from kappa fit entirely -> thermal
    if (*kappa < variable_kappa_min) *kappa = variable_kappa_min;

    //fprintf(stderr, "sigma, beta -> kappa %g %g -> %g\n", sigma, beta, *kappa);
  } else {
    *kappa = model_kappa;
  }

  double Thetae = get_model_thetae(X);
  *kappa_width = (*kappa - 3.) / *kappa * Thetae;
#if DEBUG
  if (isnan(*kappa_width) || isnan(*kappa)) {
    fprintf(stderr, "NaN kappa val! kappa, kappa_width, Thetae: %g %g %g\n", *kappa, *kappa_width, Thetae);
  }
#endif
}
