function T = addVowelsMultiHot(T)
    % Add multi-hot encoding columns for vowels (NO NONE)
    n = height(T);
    T.isA = false(n, 1);

    for i = 1:n
        v = upper(char(T.Vowels(i)));  % Convert to char and uppercase
        if contains(v, 'A'), T.isA(i) = true; end
    end
    
    fprintf('Multi-hot encoding added:\n');
    fprintf('  isA: %d samples\n', sum(T.isA));
end
