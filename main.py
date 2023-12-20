import dHexagonSentimentAnalysis as dHex
result=dHex.spam_detection('/Users/akshayv/Desktop/MLdata/sih10s.wav')
if isinstance(result, str):
    print(result)
else:
    print(f"graph coordinates->{result.coordinates}")
    print(f"caller emotions->{result.emotions}")
    print(f"call pos_percent->{result.pos_percent}")
    print(f"call neg_percent->{result.neg_percent}")
    print(f"call rating->{result.rating}")
    print(f"call language->{result.language}")
    print(f"call duration->{result.duration}")
    print(f"call transcript->{result.transcript}")
    print(f"issue ->{result.issue}")
    print(f"emotions_audio ->{result.emotions_audio}")
