function [feat_pca pca_settings] = featureReductionPCA(feat, variance_contribution)

if isempty(variance_contribution)
    variance_contribution = 0
end
[feat_pca pca_settings] = processpca(feat', 'maxfrac', variance_contribution);
