function setCollapseExpandIcon(button) {
	// Set the collapse expand button to the correct icon
	$(button).removeClass('far fa-minus-square')
	$(button).removeClass('fas fa-minus-square')
	$(button).removeClass('far fa-plus-square')
	$(button).removeClass('fas fa-plus-square')
	if ($(button).closest('.input_area').hasClass('collapsed')) {
		if ($(button).is(':hover')){$(button).addClass('fas fa-plus-square')} 
		else {$(button).addClass('far fa-plus-square')}
	} else { 
		if ($(button).is(':hover')){$(button).addClass('fas fa-minus-square')}
		else {$(button).addClass('far fa-minus-square')}
	}
}

$(document).ready(function(){
	// Collapse input_area and change icon upon click
	$('.collapse_expand_button').click(function(){
		$(this).closest('.input_area').toggleClass('collapsed')
		setCollapseExpandIcon(this)
	});
	// Highlight icon upon hover
	$('.collapse_expand_button').hover(
		function(){setCollapseExpandIcon(this)},
		function(){setCollapseExpandIcon(this)}
	)
	// Run on each
	$('.collapse_expand_button').each(function(){setCollapseExpandIcon(this)});
})