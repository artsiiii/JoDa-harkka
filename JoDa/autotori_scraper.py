# -*- coding: utf-8 -*-
import scrapy
import time
import requests


class AutotoriScraperSpider(scrapy.Spider):
    name = "autotori_scraper"
    allowed_domains = ["autot.tori.fi"]
    next_page_url = 'https://autot.tori.fi/vaihtoautot?ajoneuvotyyppi=henkiloauto'
    
    start_urls = [next_page_url,]

    def parse(self, response):
       
        all_data = response.css(".w-full.p-2")
        car_title = all_data.css('[data-testid="adCardTitle"]')
        car_description = all_data.css('.mb-2')
        car_price = all_data.css('.m\:mb-4.typography_shared__c9KpG.typography_m-headingS__2t2I5.typography_subtitle2__1IqnW')
        car_data = all_data.css('[data-testid="adCardDetailsMobile"]')
        car_city_date = all_data.css('.mb-1.text-gray-dark.truncate.typography_shared__c9KpG.typography_m-body1__2QrQ7.typography_body2__25hFD')
        
        
        for i in range(len(car_title)):
            car_title[i] = "".join(car_title[i].css('::text').extract())
            car_description[i] = "".join(car_description[i].css('::text').extract())
            car_price[i] ="".join(car_price[i].css('::text').extract())
            car_data[i] ="".join(car_data[i].css('::text').extract())
            car_city_date[i] ="".join(car_city_date[i].css('::text').extract())
           
            
                                                       
        for i in range(len(car_title)):                        
            car = {
                'response' : response,
                'title' : car_title[i],
                'description' : car_description[i],
                'price' : car_price[i],                
                'data' : car_data[i],
                'city, date & time' : car_city_date[i]                
            }
            yield car
            
        next_page_url = response.css('a.pagination_next__9Oo3k ::attr(href)').extract_first()
        if next_page_url is not None:
            if requests.get(response.urljoin(next_page_url)) != '<Response [200]>':
                time.sleep(1)
                next_page_url = response.css('a.pagination_next__9Oo3k ::attr(href)').extract_first()
                
            yield scrapy.Request(response.urljoin(next_page_url))
            #yield response.follow(next_page_url, callback=self.parse)
