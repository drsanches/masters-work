package dataset_creator.feature_extractor;

public enum FeatureName {

    AVG_TIME_BETWEEN_TRANS,
    AVG_TIME_BETWEEN_SENT_TRANS,
    AVG_TIME_BETWEEN_RECEIVED_TRANS,
    DEVIATION_TIME_BETWEEN_TRANS,
    DEVIATION_TIME_BETWEEN_SENT_TRANS,
    DEVIATION_TIME_BETWEEN_RECEIVED_TRANS,

    AVG_TRANS_ETH,
    AVG_ETH_SENT,
    AVG_ETH_RECEIVED,
    DEVIATION_TRANS_ETH,
    DEVIATION_ETH_SENT,
    DEVIATION_ETH_RECEIVED,

    PERCENT_OF_SMART_CONTRACT_TRANS,
    PERCENT_OF_TRANS_RECEIVED_FROM_SMART_CONTRACTS,
    PERCENT_OF_TRANS_SENT_TO_SMART_CONTRACTS,

    PERCENT_OF_SMART_CONTRACT_ETH,
    PERCENT_OF_ETH_RECEIVED_FROM_SMART_CONTRACTS,
    PERCENT_OF_ETH_SENT_TO_SMART_CONTRACTS
}